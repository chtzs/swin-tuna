_base_ = [
    './swin_tuna_sam_fpn_based.py', '../datasets/foodseg103_640x640.py',
    '../default_runtime.py', '../schedule_100k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
checkpoint_file = "pretrain/swin_large_semantic_sam.pth"
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformerTunaSAM',
        pretrain_img_size=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12),
    decode_head=dict(
        num_classes=104
    )
)