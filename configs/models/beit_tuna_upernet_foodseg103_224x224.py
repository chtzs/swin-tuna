_base_ = [
    './swin_tuna_fpn_based.py', '../datasets/foodseg103_224x224.py',
    '../default_runtime.py', '../schedule_100k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (224, 224)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='BEiTTuna',
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23],
    ),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        num_classes=104,
        channels=1024,
    ),
    auxiliary_head=dict(
        in_channels=1024,
        num_classes=104
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(426, 426))
)