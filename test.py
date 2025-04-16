# from swin_tuna.modeling.beit import BEiT
import torch
import time
from functools import partial

from swin_tuna.modeling.vit import ImageEncoderViT
# beit = BEiT(
#     img_size=224,
#     patch_size=16,
#     embed_dim=1024,
#     depth=24,
#     num_heads=16,
#     mlp_ratio=4,
#     qkv_bias=True,
#     use_abs_pos_emb=False,
#     use_rel_pos_bias=True,
#     init_values=1e-6,
#     drop_path_rate=0.2,
#     out_indices=[7, 11, 15, 23],
# ).cuda()
# beit.eval()
# beit.init_weights("/data/diskb/quyang/haotian/beit_large_patch16_224_pt22k_ft22k.pth")
prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16

model = ImageEncoderViT(
    depth=12,
    embed_dim=768,
    img_size=image_size,
    mlp_ratio=4,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    num_heads=12,
    patch_size=vit_patch_size,
    qkv_bias=True,
    use_rel_pos=True,
    global_attn_indexes=[2, 5, 8, 11],
    window_size=14,
    out_chans=256,
    use_abs_pos=False,
).cuda()
model.eval()

def load_checkpoint():
    state_dict = torch.load('pretrain/sam_vit_b_01ec64.pth')
    re = {}
    for k, v in state_dict.items():
        if k.startswith("image_encoder"):
            re[k.replace("image_encoder.", "")] = v

    return re

model.load_state_dict(load_checkpoint(), strict=False)
print("Done.")
# exit()
for i in range(10):
    a = torch.rand((4, 3, 384, 384)).cuda()
    b = model(a)
    re = [x.shape for x in b]
    print(re)
time.sleep(10)
