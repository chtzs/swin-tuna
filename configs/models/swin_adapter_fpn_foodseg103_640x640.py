_base_ = [
    './swin_large_fpn_based.py', '../datasets/foodseg103_640x640.py',
    '../default_runtime.py', '../schedule_100k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformerAdapter'
    ),
    decode_head=dict(
        num_classes=104
    )
)