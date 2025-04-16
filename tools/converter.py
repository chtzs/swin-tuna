import torch
# from mmseg.registry import MODELS


# def load_mmseg_swin():       
#     config = dict(
#         type='SwinTransformer',
#         pretrain_img_size=384,
#         embed_dims=192,
#         mlp_ratio=4,
#         depths=[2, 2, 18, 2],
#         num_heads=[6, 12, 24, 48],
#         strides=(4, 2, 2, 2),
#         out_indices=(0, 1, 2, 3),
#         window_size=12,
#         use_abs_pos_embed=False,
#         drop_path_rate=0.3,
#         patch_norm=True,
#         act_cfg=dict(type='GELU')
#     )
#     model = MODELS.build(config)
#     return model

def convert_semantic_sam(in_path="pretrain/swinl_only_sam_many2many.pth", out_path="pretrain/swin_large_semantic_sam.pth"):
    state_dict = torch.load("pretrain/swinl_only_sam_many2many.pth")

    modified = {}
    for k, v in state_dict['model'].items():
        if not "model.backbone" in k: continue
        new_key = k.replace("model.backbone.", "")
        new_key = new_key.replace("layers", "stages")
        new_key = new_key.replace("patch_embed.proj", "patch_embed.projection")
        new_key = new_key.replace("attn.", "attn.w_msa.")
        new_key = new_key.replace("mlp.fc1", "ffn.layers.0.0")
        new_key = new_key.replace("mlp.fc2", "ffn.layers.1")
            
        modified[new_key] = v

    torch.save(modified, out_path)

convert_semantic_sam()