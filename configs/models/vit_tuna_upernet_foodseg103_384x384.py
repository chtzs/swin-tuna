_base_ = [
    './vit_tuna_upernet_based.py', '../datasets/foodseg103_384x384.py',
    '../default_runtime.py', '../schedule_100k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (384, 384)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=104
    ),
    # auxiliary_head=dict(
    #     num_classes=104
    # )
)