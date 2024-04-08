import torch

def apply_boxes(resizer, tree_boxes, building_boxes, original_size):
    boxes = torch.cat((tree_boxes, building_boxes), dim=0)
    return resizer.apply_boxes_torch(boxes= boxes, original_size = original_size)

def separate_classes(masks: torch.Tensor, num_trees: int ):
    """
    Args:
        masks (torch.tensor): shape (B, H, W)
    """
    
    max_tree_masks, _ = torch.max(masks[:num_trees], dim=0) #(H, W)
    max_building_masks, _ = torch.max(masks[num_trees:], dim=1) #(H, W)
    stacked_masks = torch.stack((max_tree_masks, max_building_masks), dim=0)  # shape (2, H, W)
    return stacked_masks
    
    