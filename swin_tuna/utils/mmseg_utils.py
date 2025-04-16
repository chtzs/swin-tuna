
from collections import defaultdict
from typing import Optional, Union, Sequence

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.registry import DefaultScope
from mmseg.models import BaseSegmentor
from mmseg.apis import init_model, inference_model
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.models.utils import resize
from mmseg.registry import MODELS, DATASETS, TRANSFORMS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from PIL import Image


ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def load_model(model_config: str, model_checkpoint: str) -> BaseSegmentor:
    """Load model from config

    Args:
        model_config (str): Model config path
        model_checkpoint (str): Model checkpoint path

    Returns:
        BaseSegmentor: Model
    """
    model = init_model(model_config, 
        checkpoint=model_checkpoint, 
        device='cuda:0')
    return model

def load_dataloader(dataloader_config='/data/diskb/quyang/haotian/MyNet/configs/uecfoodpix_complete_384x384.py') -> data.DataLoader:
    dataloader_dict = Config.fromfile(dataloader_config)
    # Build dataloader from validation dataset
    dataloader = Runner.build_dataloader(
        dataloader=dataloader_dict['val_dataloader'],
        seed=12345678
    )
    return dataloader

def _preprare_data(imgs, model):
    cfg = model.cfg
    for t in cfg.test_pipeline:
        # Modified here
        if t.get('type').find('LoadAnnotations') != -1:
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch

def inference_model(model: BaseSegmentor, img: ImageType):
    """Inference image(s) with the segmentor.
    
    CAUTION: Modified from mmseg.api.inference_model

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        :obj:`SegDataSample` or list[:obj:`SegDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the segmentation results directly.
    """
    # prepare data
    data, is_batch = _preprare_data(img, model)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]

def sem_seg_postprocess(seg_logits: torch.Tensor, data_sample: SegDataSample, align_corners=True) -> torch.Tensor:
    _, _, H, W = seg_logits.shape
    img_meta = data_sample.metainfo
    # remove padding area
    if 'img_padding_size' not in img_meta:
        padding_size = img_meta.get('padding_size', [0] * 4)
    else:
        padding_size = img_meta['img_padding_size']
    padding_left, padding_right, padding_top, padding_bottom =\
        padding_size
    # i_seg_logits shape is 1, C, H, W after remove padding
    i_seg_logits = seg_logits[...,
                                padding_top:H - padding_bottom,
                                padding_left:W - padding_right]

    flip = img_meta.get('flip', None)
    if flip:
        flip_direction = img_meta.get('flip_direction', None)
        assert flip_direction in ['horizontal', 'vertical']
        if flip_direction == 'horizontal':
            i_seg_logits = i_seg_logits.flip(dims=(3, ))
        else:
            i_seg_logits = i_seg_logits.flip(dims=(2, ))

    # resize as original shape
    i_seg_logits = resize(
        i_seg_logits,
        size=img_meta['ori_shape'],
        mode='bilinear',
        align_corners=align_corners,
        warning=False)
    return i_seg_logits

def add_custom_module():
    MODELS._locations.append('swin_tuna.mmseg')
    DATASETS._locations.append('swin_tuna.mmseg')
    TRANSFORMS._locations.append('swin_tuna.mmseg')