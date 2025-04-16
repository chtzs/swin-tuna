_base_ = [
    './swin_mona_fpn_based.py', '../datasets/foodseg103_640x640.py',
    '../default_runtime.py', '../schedule_100k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        num_classes=104
    )
)