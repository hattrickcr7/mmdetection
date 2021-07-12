import os.path
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from xml.dom.minidom import Document


from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    pseudo_th=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']
    for i, data in enumerate(data_loader):
        tags = data['gt_labels']
        data.pop('gt_labels')
        data.pop('gt_bboxes')
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        for idx, (d, tags_img, r) in enumerate(zip(data['img_metas'], tags, result)):
            doc = Document()
            annotation = doc.createElement("annotation")
            doc.appendChild(annotation)
            folder = doc.createElement("folder")
            filename = doc.createElement("filename")
            size = doc.createElement("size")
            width = doc.createElement("width")
            height = doc.createElement("height")
            depth = doc.createElement("depth")
            size.appendChild(width)
            size.appendChild(height)
            size.appendChild(depth)
            annotation.appendChild(folder)
            annotation.appendChild(filename)
            annotation.appendChild(size)
            filenamee = doc.createTextNode(d.data[0][0]['ori_filename'].split('/')[1])
            foldername = doc.createTextNode(d.data[0][0]['ori_filename'].split('/')[0])
            widthnum = doc.createTextNode(str(d.data[0][0]['ori_shape'][0]))
            heightnum = doc.createTextNode(str(d.data[0][0]['ori_shape'][1]))
            depthnum = doc.createTextNode(str(d.data[0][0]['ori_shape'][2]))
            folder.appendChild(foldername)
            filename.appendChild(filenamee)
            width.appendChild(widthnum)
            height.appendChild(heightnum)
            depth.appendChild(depthnum)
            for t in tags_img[0]:
                tag = doc.createElement("tag")
                tagname = doc.createTextNode(VOC_CLASSES[t])
                tag.appendChild(tagname)
                annotation.appendChild(tag)
            for ridx in range(len(r)):
                classname = VOC_CLASSES[ridx]
                for box in r[ridx]:
                    if box[4] < pseudo_th:
                        continue
                    objectt = doc.createElement("object")
                    annotation.appendChild(objectt)
                    bndbox = doc.createElement("bndbox")
                    objectt.appendChild(bndbox)
                    xmin = doc.createElement("xmin")
                    ymin = doc.createElement("ymin")
                    xmax = doc.createElement("xmax")
                    ymax = doc.createElement("ymax")
                    score = doc.createElement("score")
                    calss = doc.createElement("name")
                    objectt.appendChild(calss)
                    calss.appendChild(doc.createTextNode(classname))
                    bndbox.appendChild(xmin)
                    bndbox.appendChild(ymin)
                    bndbox.appendChild(xmax)
                    bndbox.appendChild(ymax)
                    bndbox.appendChild(score)
                    difficult = doc.createElement("difficult")
                    difficult.appendChild(doc.createTextNode('0'))
                    objectt.appendChild(difficult)
                    xminnum = doc.createTextNode(str(box[0]))
                    yminnum = doc.createTextNode(str(box[1]))
                    xmaxnum = doc.createTextNode(str(box[2]))
                    ymaxnum = doc.createTextNode(str(box[3]))
                    scorenum = doc.createTextNode(str(box[4]))
                    xmin.appendChild(xminnum)
                    ymin.appendChild(yminnum)
                    xmax.appendChild(xmaxnum)
                    ymax.appendChild(ymaxnum)
                    score.appendChild(scorenum)
            f = open(os.path.join('data\\VOCdevkit', os.path.join(d.data[0][0]['filename'].split('/')[2], os.path.join('pseudo_label', d.data[0][0]['ori_filename'].split('/')[1].split('.')[0]+'.xml'))), "w")
            f.write(doc.toprettyxml(indent="  "))
            f.close()

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
