# dataset settings
dataset_type = 'UECFoodPixDataset'
data_root = '/root/workspace/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE'
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFilePillow'),
    dict(type='LoadAnnotationsCustom', channel='r'),
    # dict(
    #     type='Resize',
    #     scale=crop_size,
    #     keep_ratio=True),
    dict(
        type='RandomResize',
        scale=(384 * 4, 384),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFilePillow'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotationsCustom', channel='r'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/mask'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/img',
            seg_map_path='test/mask'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# val_evaluator = dict(type='NoLabelIoUMetric')
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
