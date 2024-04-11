import torch
import numpy as np

def apply_boxes(resizer, tree_boxes, building_boxes, original_size):
    boxes = torch.cat((tree_boxes, building_boxes), dim=0)
    return resizer.apply_boxes_torch(boxes= boxes, original_size = original_size)

def separate_classes_tensor(masks: torch.Tensor , num_trees: int ) -> torch.Tensor:
    """
    Separates the masks into two classes: trees and buildings using PyTorch.
    Takes a 3D tensor of masks and an integer indicating the number of tree masks.

    Args:
        masks (torch.Tensor): A 3D tensor of shape (B, H, W), where B is the number of masks
        num_trees (int): The number of masks that correspond to trees.

    Returns:
        torch.Tensor: A 3D tensor of shape (2, H, W), first channel tree class, second channel build class
    """
    if num_trees == 0: #no tree masks
        tree_mask = torch.full_like(masks[0], fill_value=float('-inf'), device=masks.device)
    else:
        tree_mask = torch.max(masks[:num_trees], dim=0).values
        
    if num_trees == masks.shape[0]: #no building masks
        building_mask = torch.full_like(masks[0], fill_value=float('-inf'), device=masks.device)
    else:
        building_mask = torch.max(masks[num_trees:], dim=0).values
        
    stacked_masks = torch.stack((tree_mask, building_mask), dim=0)  # shape (2, H, W)
    return stacked_masks

def collapse_masks_tensor(masks: torch.Tensor , num_trees: int ) -> torch.Tensor:
    """
    Separates the masks into two classes: trees and buildings using PyTorch.
    Takes a 3D tensor of masks and an integer indicating the number of tree masks.

    Args:
        masks (torch.Tensor): A 3D tensor of shape (B, H, W), where B is the number of masks
        num_trees (int): The number of masks that correspond to trees.

    Returns:
        torch.Tensor: A 3D tensor of shape (2, H, W), first channel tree class, second channel build class
    """
    if num_trees == 0: #no tree masks
        tree_mask = torch.full_like(masks[0], fill_value=float('-inf'), device=masks.device)
    else:
        tree_mask = torch.max(masks[:num_trees], dim=0).values
        
    if num_trees == masks.shape[0]: #no building masks
        building_mask = torch.full_like(masks[0], fill_value=float('-inf'), device=masks.device)
    else:
        building_mask = torch.max(masks[num_trees:], dim=0).values
        
    stacked_masks = torch.stack((tree_mask, building_mask), dim=0)  # shape (2, H, W)
    return stacked_masks

def separate_classes_numpy(masks: np.ndarray , num_trees: int ) -> np.ndarray:
    """
    Separates the masks into two classes: trees and buildings using PyTorch.
    Takes a 3D tensor of masks and an integer indicating the number of tree masks.

    Args:
        masks (torch.Tensor): A 3D tensor of shape (B, H, W), where B is the number of masks
        num_trees (int): The number of masks that correspond to trees.

    Returns:
        torch.Tensor: A 3D tensor of shape (2, H, W), first channel tree class, second channel build class
    """
    if num_trees == 0: #no tree masks
        tree_mask = np.full(masks[0].shape, fill_value=float('-inf'), dtype=np.float32)
    else:
        tree_mask = np.max(masks[:num_trees], axis=0)
        
    if num_trees == masks.shape[0]: #no building masks
        building_mask = np.full(masks[0].shape, fill_value=float('-inf'), dtype=np.float32)
    else:
        building_mask = np.max(masks[num_trees:], axis=0)
        
    stacked_masks = np.stack((tree_mask, building_mask), axis=0)  # shape (2, H, W)
    return stacked_masks

def placeholder_masks(original_img_h_w,
                    device,
                    type: str):
    if type == 'torch':
        all_masks = torch.empty((0, *original_img_h_w),
                                dtype=torch.float32,
                                device = device)
    elif type == 'numpy':
            all_masks = np.empty((0, *original_img_h_w),
                                dtype=np.float32)
            
    return all_masks
