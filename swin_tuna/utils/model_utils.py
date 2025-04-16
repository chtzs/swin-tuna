import torch
import torch.nn as nn
import time
from typing import List, Dict

class Stopwatcher():
    def __init__(self):
        self.start_time = time.time()
        
    def begin(self):
        self.start_time = time.time()
        
    def end(self, name="Test"):
        end_time = time.time()
        print(f'Time cost of {name}: {end_time - self.start_time}')

def get_stopwatcher():
    stopwatcher = Stopwatcher()
    return stopwatcher.begin, stopwatcher.end

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
        
def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
        
def print_result(result: dict):
    for k, v in result.items():
        if isinstance(v, list):
            print(f'{k} is a list: ')
        elif isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")

def get_binary_masks(semantic_mask: torch.Tensor, num_classes: int, reduce_zero_labels=False) -> torch.Tensor:
    if reduce_zero_labels:
        semantic_mask[semantic_mask == 255] = num_classes
        num_classes += 1
    y, x = semantic_mask.shape
    target_onehot = torch.zeros(num_classes, y, x, device=semantic_mask.device)
    target_onehot = target_onehot.scatter(dim=0, index=semantic_mask.unsqueeze(0), value=1)
    if reduce_zero_labels:
        target_onehot = target_onehot[:-1]
    return target_onehot

def batch_get_binary_masks(semantic_mask: torch.Tensor, num_classes: int, reduce_zero_labels=False) -> torch.Tensor:
    if len(semantic_mask.shape) == 2:
        semantic_mask = semantic_mask.unsqueeze(0)
    if reduce_zero_labels:
        semantic_mask[semantic_mask == 255] = num_classes
        num_classes += 1
    batch_size, y, x = semantic_mask.shape
    target_onehot = torch.zeros(batch_size, num_classes, y, x, device=semantic_mask.device)
    target_onehot = target_onehot.scatter(dim=1, index=semantic_mask.unsqueeze(1), value=1)
    if reduce_zero_labels:
        target_onehot = target_onehot[:, :-1, ...]
    return target_onehot
    
def get_labels(num_classes: int, device) -> torch.Tensor:
    # Create identity matrix
    labels = torch.eye(num_classes, dtype=torch.int64, device=device)
    return labels
    
def make_target_dict(ground_truth: List[torch.Tensor], num_queries: int, num_classes: int) -> List[Dict[str, torch.Tensor]]:
    results = []
    for gt in ground_truth:
        # H * W
        assert len(gt.shape) == 2
        
        gt_labels = gt.unique()
        # let E = gt_labels.shape[0]
        num_labels = gt_labels.shape[0]
        assert num_labels < num_queries, "This shouldn't be happened... You should increase number of object queries."
        h, w = gt.shape
        # NUM_CLASSES * H * W
        binary_masks = get_binary_masks(gt, num_classes, device=gt.device)
        # NUM_CLASSES * H * W -> E * H * W
        binary_masks = binary_masks[gt_labels]
        # Expand binary_masks with zero(which means negative)
        # E * H * W -> N * H * W
        binary_masks = torch.cat((binary_masks, 
                                 torch.zeros((num_queries - num_labels, h, w), device=gt.device)), 
                                 dim=0)
        
        # # NUM_CLASSES * NUM_CLASSES
        # labels = get_labels(num_classes, device=gt.device)
        # # NUM_CLASSES * NUM_CLASSES -> E * NUM_CLASSES
        # labels = labels[gt_labels]
        # # Expand labels with [1, 0, ..., 0](which means empty class)
        # empty_classes = torch.zeros(num_queries - num_labels, num_classes)
        # empty_classes[:, 0] = 1
        # # E * NUM_CLASSES -> N * NUM_CLASSES
        # labels = torch.cat((labels, 
        #                    empty_classes), 
        #                    dim=0)
        a = gt_labels.to(dtype=torch.int64)
        b = torch.zeros(num_queries - num_labels, device=gt.device)
        c = torch.cat((a, b), dim=0)
        results.append({
            # N masks, N * H * W
            'masks': binary_masks,
            # N labels, N * NUM_CLASSES
            'labels': c.to(dtype=torch.int64)
        })
    return results

# from .metric_utils import fill_zeros
def prepare_targets(ground_truth: List[torch.Tensor],  num_classes: int):
    # ground_truth: B * H * W
    results = []
    for gt in ground_truth:
        # H * W
        assert len(gt.shape) == 2
        
        labels = gt.unique()
        # 移除背景
        if labels[-1] == num_classes:
            labels = labels[:-1]
        # print(labels)
        # NUM_CLASSES * H * W
        binary_masks = get_binary_masks(gt, num_classes + 1)
        # NUM_CLASSES * H * W -> E * H * W
        binary_masks = binary_masks[labels]
        # binary_masks = fill_zeros(binary_masks, 100)
        
        results.append({
            # N masks, N * H * W
            'masks': binary_masks,
            # N labels, N 
            'labels': labels
        })
    return results

def create_targets(ground_truth: List[torch.Tensor],  num_classes: int):
    # ground_truth: B * H * W
    results = []
    for gt in ground_truth:
        # H * W
        assert len(gt.shape) == 2
        labels = gt.unique()
        # NUM_CLASSES * H * W
        binary_masks = get_binary_masks(gt, num_classes, device=gt.device)
        # NUM_CLASSES
        labels_full = torch.zeros((num_classes), device=gt.device)
        labels_full[labels] = 1
        
        results.append({
            # N masks, NUM_CLASSES * H * W
            'masks': binary_masks,
            # N labels, NUM_CLASSES 
            'labels': labels_full
        })
    return results