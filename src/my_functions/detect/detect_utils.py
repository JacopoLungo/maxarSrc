import torch
from typing import Union, List
from torchvision.ops import box_convert
import numpy as np
import groundingdino.datasets.transforms as T
from PIL import Image



def GDboxes2SamBoxes(boxes: torch.Tensor, img_shape: Union[tuple[float, float], float]):
    """
    Convert the boxes from the format cxcywh to the format xyxy.
    Inputs:
        boxes: torch.Tensor of shape (N, 4). Where boxes are in the format cxcywh (the output of GroundingDINO).
        img_shape: tuple (h, w)
        img_res: float, the resolution of the image (mt/pxl).
    Output:
        boxes: torch.Tensor of shape (N, 4). Where boxes are in the format xyxy.
    """
    if isinstance(img_shape, (float, int)):
        img_shape = (img_shape, img_shape)
    
    h, w =  img_shape
    SAM_boxes = boxes.clone()
    SAM_boxes = SAM_boxes * torch.Tensor([w, h, w, h])
    SAM_boxes = box_convert(boxes=SAM_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return SAM_boxes

def GD_img_load(np_img_rgb: np.array)-> torch.Tensor:
    """
    Transform the image from np.array to torch.Tensor and normalize it.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(np_img_rgb)
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed

def filter_on_box_area_mt2(boxes, img_shape: Union[tuple[float, float], float], img_res, min_area_mt2 = 0, max_area_mt2 = 1500):
    """
    Filter boxes based on min and max area.
    Inputs:
        boxes: torch.Tensor of shape (N, 4). Where boxes are in the format cxcywh (the output of GroundingDINO).
        img_shape: tuple (h, w)
        img_res: float, the resolution of the image (mt/pxl).
        min_area_mt2: float
        max_area_mt2: float
    Output:
        keep_ix: torch.Tensor of shape (N,)
    """
    if isinstance(img_shape, (float, int)):
        img_shape = (img_shape, img_shape)
    
    h, w =  img_shape
    tmp_boxes = boxes.clone()
    tmp_boxes = tmp_boxes * torch.Tensor([w, h, w, h])

    area_mt2 = torch.prod(tmp_boxes[:,2:], dim=1) * img_res**2
    keep_ix = (area_mt2 > min_area_mt2) & (area_mt2 < max_area_mt2)
    
    return keep_ix

def filter_on_box_ratio(boxes, min_edges_ratio = 0):
    keep_ix = (boxes[:,2] / boxes[:,3] > min_edges_ratio) & (boxes[:,3] / boxes[:,2] > min_edges_ratio)
    return keep_ix

def reduce_tree_boxes(boxes, reduce_perc):
    """
    Reduce the size of the boxes by 10%. Keeping the center fixed.
    Input:
        boxes: torch.Tensor of shape (N, 4). Where boxes are in the format cxcywh (the output of GroundingDINO).
        reduce_perc: float, the percentage to reduce the boxes.
    Output:
        boxes: torch.Tensor of shape (N, 4). Where reduced boxes are in the format cxcywh.
    """
    reduced_boxes = boxes.clone()
    reduced_boxes[:,2:] = reduced_boxes[:,2:] * (1 - reduce_perc)
    return reduced_boxes