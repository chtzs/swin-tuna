import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple
from PIL import Image
from random import randrange

def get_rand_colormap(size: int):
    def get_rand_color():
        return [randrange(0, 256), randrange(0, 256), randrange(0, 256), int(255)]
    cmap = {}
    for i in range(size):
        cmap[i] = get_rand_color()
        
    cmap[255] = get_rand_color()
        
    return cmap

def read_img(img_path: str) -> np.ndarray:
    img = Image.open(img_path)
    img_array = np.asarray(img)\
            .squeeze().transpose((1, 0, -1)).astype(np.uint8)
    return img_array

# 假设mask是一个PyTorch张量,尺寸为W * H
def draw_mask(img_path: str, mask: Union[torch.Tensor, np.ndarray], output_path: str, color_map: dict, reduce_zero_label=False, alpha=0.75):
    # 读取原图
    img = Image.open(img_path)
    img_array = np.asarray(img)\
            .squeeze().transpose((1, 0, -1)).astype(np.uint8)
    img_width, img_height = img.size
    # 如果不是RGBA图片，就拼接一个通道
    if img_array.shape[2] < 4:
        a = img_array
        b = np.ones((img_width, img_height, 1)).astype(np.int8) * 255
        img_array = np.concatenate((a, b), axis=2)
        
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    if mask.shape != (img_width, img_height):
        print("Size is different!")
        print(mask.shape)
        print((img_width, img_height))
        exit()
        # 将mask调整为与原图尺寸一致
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img_height, img_width), mode='nearest').squeeze(0).squeeze(0)

    if reduce_zero_label:
        mask[mask == 0] = 255
        mask = mask - 1
        mask[mask == 254] = 255
    # 将mask转换为numpy array并应用颜色映射
    mask_img = np.array([color_map.get(x, x) for row in mask for x in row])
    mask_img = mask_img.reshape((*mask.shape, 4))
    
    # 使用Alpha Blending算法合并原图和mask
    blending = img_array * alpha + mask_img * (1 - alpha)
    blending = blending.astype(np.int8).transpose((1, 0, -1))
    
    result = Image.fromarray(blending, 'RGBA')
    # 保存结果为PNG图片
    result.save(output_path)
    
def postprocess_masks(
    masks: torch.Tensor,
    original_size: Tuple[int, ...],
) -> torch.Tensor:
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=True)
    return masks

def postprocess_masks2(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> torch.Tensor:
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    masks = F.interpolate(
        masks,
        (384, 384),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks

def binary_to_semantic_masks(binary_masks: torch.Tensor, mask_cls: torch.Tensor, original_size: Tuple[int, ...]) -> torch.Tensor:
    """Convert binary mask to semantic masks

    Args:
        binary_masks (torch.Tensor): B * Q * H * W
        mask_cls (torch.Tensor): B * Q * CLS_NUM

    Returns:
        torch.Tensor: semantic masks, B * H * W
    """
    # a = F.softmax(mask_cls, dim=-1)[...,1:]
    # print(torch.argmax(a, dim=-1))
    # a = F.softmax(mask_cls, dim=-1)
    # print(torch.argmax(a, dim=-1))
    # binary_masks = postprocess_masks(binary_masks, original_size)
    # # print(binary_masks)
    # binary_masks = binary_masks.sigmoid()
    # semantic_masks = torch.einsum("bqc,bqhw->bchw", mask_cls, binary_masks)
    # semantic_masks = torch.argmax(semantic_masks, dim=1)[0]
    
    binary_masks = postprocess_masks(binary_masks, original_size)
    print(binary_masks.shape)
    batch_size = binary_masks.shape[0]
    semantic_masks_list = []
    semantic_masks_list.append(binary_masks[0][101] >= 0)
    # mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
    # mask_cls = torch.argmax(mask_cls, dim=-1)
    # for b in range(batch_size):
    #     semantic_masks = torch.zeros(original_size, device=binary_masks.device)
    #     for cls_idx, binary_mask in zip(mask_cls[b], binary_masks[b]):
    #         # if cls_idx == 0: 
    #         #     continue
    #         semantic_masks[binary_mask >= 0] = cls_idx.item()
    #     semantic_masks_list.append(semantic_masks)
        
    # semantic_masks = binary_masks[0][4]
    # semantic_masks[semantic_masks >= 0] = 1
    # semantic_masks[semantic_masks < 0] = 0
    return torch.stack(semantic_masks_list)

def semantic_inference(binary_masks: torch.Tensor, mask_cls: torch.Tensor) -> torch.Tensor:
    batch_size, _, h, w = binary_masks.shape
    # mask_cls[..., -1] *= 0.1
    num_classes = mask_cls.shape[-1] - 1
    mask_cls = F.softmax(mask_cls, dim=-1) #[..., :-1]
    # cls_68 = mask_cls[..., 51, :].squeeze()
    # cls_0 = mask_cls[..., 30, :].squeeze()
    # cls_100 = mask_cls[..., 8, :].squeeze()
    # cls_35 = mask_cls[..., 36, :].squeeze()
    # print(68)
    # print(cls_68.max())
    # print(cls_68[68])
    # print("35")
    # print(cls_35.max())
    # print(cls_35[35])
    # print(cls_35[100])
    # print(0)
    # print(cls_0.max())
    # print(cls_0[0])
    # print(100)
    # print(cls_100.max())
    # print(cls_100[100])
    mask_cls = torch.argmax(mask_cls, dim=-1)
    semantic_masks_list = []
    # print(mask_cls)
    for b in range(batch_size):
        semantic_masks = torch.zeros((h, w), device=binary_masks.device)
        for cls_idx, binary_mask in zip(mask_cls[b], binary_masks[b]):
            if cls_idx == num_classes: 
                continue
            # if cls_idx == 0 or cls_idx == 41 or cls_idx == 101:
            semantic_masks[binary_mask >= 0] = cls_idx.item() + 1
        semantic_masks_list.append(semantic_masks)
    return torch.stack(semantic_masks_list)

def semantic_inference2(binary_masks: torch.Tensor, mask_cls: torch.Tensor) -> torch.Tensor:
    empty_threshold = 0.75
    batch_size, _, h, w = binary_masks.shape
    num_classes = mask_cls.shape[-1] - 1
    mask_cls = F.softmax(mask_cls, dim=-1)
    semantic_masks_list = []
    for b in range(batch_size):
        semantic_masks = torch.zeros((h, w, num_classes + 1), device=binary_masks.device)
        semantic_masks[..., -1] = empty_threshold
        # N masks and classes
        for cls_logits, binary_mask in zip(mask_cls[b], binary_masks[b]):
            non_empty_logits = torch.zeros_like(cls_logits, device=binary_masks.device)
            non_empty_logits[:-1] = cls_logits[:-1]
            empty_logits = torch.zeros_like(cls_logits, device=binary_masks.device)
            empty_logits[-1] = cls_logits[-1]
            
            # semantic_masks[binary_mask < 0] += empty_logits
            semantic_masks[binary_mask >= 0] += non_empty_logits
        semantic_masks = semantic_masks.argmax(dim=-1, keepdim=True).squeeze()
        semantic_masks += 1
        semantic_masks[semantic_masks == num_classes + 1] = 0
        semantic_masks_list.append(semantic_masks)
    return torch.stack(semantic_masks_list)
        
        
    