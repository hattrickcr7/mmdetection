_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_voc_v2.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/voc_stage2.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
