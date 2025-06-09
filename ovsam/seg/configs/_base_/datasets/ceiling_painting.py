from mmcv import LoadImageFromFile, RandomResize
from mmdet.datasets import AspectRatioBatchSampler
from mmdet.datasets.transforms import LoadAnnotations, Resize, RandomFlip, PackDetInputs, RandomCrop
from mmengine.dataset import DefaultSampler

from seg.datasets.ceiling_painting import CeilingPaintingSegmentationDataset
from seg.datasets.pipeliens.loading import FilterAnnotationsHB
from seg.evaluation.ins_cls_iou_metric import InsClsIoUMetric

from ext.class_names.ceiling_ids import CEILING_BASE_IDS, CEILING_NOVEL_IDS, CEILING_ALL_IDS

# Dataset settings
dataset_type = CeilingPaintingSegmentationDataset
data_root = '/home/ra78lof/consulting_pro/ceiling_easy_train_with_masks/'  
backend_args = None

# Define Classes
class_names = ['#', 'brief', 'mural', 'relief']

image_size = (1024, 1024)

# Data pipeline for training
train_pipeline = [
    dict(type=LoadImageFromFile, to_float32=True, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True, backend_args=backend_args),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomResize,
        resize_type=Resize,
        scale=image_size,
        ratio_range=(.9, 2.),
        keep_ratio=True,
    ),
    dict(
        type=RandomCrop,
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type=FilterAnnotationsHB,
        by_box=False,
        by_mask=True,
        min_gt_mask_area=32,
    ),
    dict(type=PackDetInputs)
]

# Data pipeline for testing/inference
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1024, 1024), keep_ratio=True),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Training dataset
train_dataloader = dict(
    batch_size=2,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='',  # Leave empty if using directory structure
        data_prefix=dict(img='img_dir/', seg='ann_dir/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

# Validation dataset
val_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='', # Leave empty if using directory structure
        data_prefix=dict(img='img_dir/', seg='ann_dir/'),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# Test dataset
test_dataloader = val_dataloader

# Evaluator
val_evaluator = [
    dict(
        type=InsClsIoUMetric,
        prefix='ceiling_ins',
        base_classes=CEILING_BASE_IDS,
        novel_classes=CEILING_NOVEL_IDS,
    ),
]
test_evaluator = val_evaluator

# For zero-shot inference if needed
zero_shot_dataset = dict(
    type='CustomZeroShotDataset',
    data_root=data_root,
    custom_classes=['#','brief','mural','relief'],  
    data_prefix=dict(img='img_dir', seg='ann_dir'),
    test_mode=True,
    pipeline=test_pipeline)