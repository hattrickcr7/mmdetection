import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule

from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
import torch


@HEADS.register_module()
class UFOWeakHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 in_channels=256,
                 with_ref=True,
                 with_bbox_pred=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 loss_img=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 *args,
                 **kwargs):
        super(UFOWeakHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_ref = with_ref
        self.with_bbox_pred = with_bbox_pred
        self.loss_img = build_loss(loss_img)
        self.in_channels = in_channels

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.in_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.in_channels)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            self.fc_det = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)

        if self.with_ref:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.ref_cls1 = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            self.ref_cls2 = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            self.ref_cls3 = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_bbox_pred:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.bbox_reg1 = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
            self.bbox_reg2 = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
            self.bbox_reg3 = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        det_score = self.fc_det(x_reg) if self.with_reg else None

        ref1_logit = self.ref_cls1(x)
        bbox_pred1 = self.bbox_reg1(x)
        # ref2_logit = self.ref_cls2(x)
        # bbox_pred2 = self.bbox_reg2(x)
        # ref3_logit = self.ref_cls3(x)
        # bbox_pred3 = self.bbox_reg3(x)

        # ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        # bbox_preds = [bbox_pred1, bbox_pred2, bbox_pred3]
        ref_logits = ref1_logit
        bbox_preds = bbox_pred1

        return cls_score, det_score, ref_logits, bbox_preds

    @force_fp32(apply_to=('cls_score', 'det_score', 'ref_logits', 'ref_bbox_preds'))
    def loss(self,
             bbox_results,
             gt_bboxes,
             gt_labels,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             epsilon=1e-10):
        losses = dict()
        assert bbox_results is not None
        cls_score = bbox_results['cls_score']
        det_score = bbox_results['det_score']
        ref_logits = bbox_results['ref_logits']
        ref_bbox_preds = bbox_results['ref_bbox_preds']
        cls_score = F.softmax(cls_score, dim=1)

        det_score_list = det_score.split([p.bboxes.shape[0] for p in rois])
        final_det_score = []
        for det_score_per_imgs in det_score_list:
            det_score_per_imgs = F.softmax(det_score_per_imgs, dim=0)
            final_det_score.append(det_score_per_imgs)
        final_det_score = torch.cat(final_det_score, dim=0)

        device = cls_score.device
        num_classes = cls_score.shape[1]

        final_score = cls_score * final_det_score
        final_score_list = final_score.split([p.bboxes.shape[0] for p in rois])
        # ref_logits = [rs.split([p.bboxes.shape[0] for p in rois]) for rs in ref_logits]
        # ref_bbox_preds = [rbp.split([p.bboxes.shape[0] for p in rois]) for rbp in ref_bbox_preds]
        ref_logits = ref_logits.split([p.bboxes.shape[0] for p in rois])
        ref_bbox_preds = ref_bbox_preds.split([p.bboxes.shape[0] for p in rois])

        losses['loss_img'] = 0
        losses['loss_cls'] = 0
        losses['loss_bbox'] = 0
        for idx, (final_score_per_im, targets_per_im, proposals_per_image, ref_logit, ref_bbox_pred) in enumerate(zip(final_score_list, gt_labels, rois, ref_logits, ref_bbox_preds)):
            labels_per_im = generate_img_label(num_classes, targets_per_im, device)
            score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            loss_img = self.loss_img(score_per_im, labels_per_im)
            losses['loss_img'] += loss_img

            if ref_logit is not None:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                if ref_logit.numel() > 0:
                    loss_cls_ = self.loss_cls(
                        ref_logit,
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                    if isinstance(loss_cls_, dict):
                        losses.update(loss_cls_)
                    else:
                        losses['loss_cls'] = loss_cls_
                    if self.custom_activation:
                        acc_ = self.loss_cls.get_accuracy(ref_logit, labels)
                        losses.update(acc_)
                    else:
                        losses['acc'] = accuracy(ref_logit, labels)
            if ref_bbox_pred is not None:
                bg_class_ind = self.num_classes
                # 0~self.num_classes-1 are FG, self.num_classes is BG
                pos_inds = (labels >= 0) & (labels < bg_class_ind)
                # do not perform bounding box regression for BG anymore.
                if pos_inds.any():
                    if self.reg_decoded_bbox:
                        # When the regression loss (e.g. `IouLoss`,
                        # `GIouLoss`, `DIouLoss`) is applied directly on
                        # the decoded bounding boxes, it decodes the
                        # already encoded coordinates to absolute format.
                        ref_bbox_pred = self.bbox_coder.decode(rois[:, 1:], ref_bbox_pred)
                    if self.reg_class_agnostic:
                        pos_bbox_pred = ref_bbox_pred.view(
                            ref_bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                    else:
                        pos_bbox_pred = ref_bbox_pred.view(
                            ref_bbox_pred.size(0), -1,
                            4)[pos_inds.type(torch.bool),
                               labels[pos_inds.type(torch.bool)]]
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                else:
                    losses['loss_bbox'] = ref_bbox_pred[pos_inds].sum()

        return losses

def generate_img_label(num_classes, labels, device):
    img_label = torch.zeros(num_classes)
    img_label[labels.long()] = 1
    img_label[0] = 0
    return img_label.to(device)