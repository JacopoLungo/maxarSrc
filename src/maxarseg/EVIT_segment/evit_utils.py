import torch

def apply_boxes(resizer, tree_boxes, building_boxes, original_size):
    boxes = torch.cat((tree_boxes, building_boxes), dim=0)
    return resizer.apply_boxes_torch(boxes= boxes, original_size = original_size)

def separate_classes(masks: torch.Tensor , num_trees: int ):
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