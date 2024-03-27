
from pathlib import Path
import os
import sys

from groundingdino.util.inference import load_model as GD_load_model
sys.path.append('/home/vaschetti/maxarSrc/models/EfficientSAM')
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt


class SegmentConfig:
    """
    Config class for the segmentation pipeline.
    It contains detection and segmentation parameters as well as the models themselves. 
    """
    def __init__(self,
                 batch_size,
                 size = 600,
                 stride = 400,
                 
                 device = 'cuda',
                 GD_root = "/home/vaschetti/maxarSrc/models/GDINO",
                 GD_config_file = "GroundingDINO_SwinT_OGC.py",
                 GD_weights = "groundingdino_swint_ogc.pth",
                 
                 TEXT_PROMPT = 'bush', #'green tree'
                 BOX_THRESHOLD = 0.15,
                 TEXT_THRESHOLD = 0.30,
                 
                 max_area_GD_boxes_mt2 = 6000,
                 min_ratio_GD_boxes_edges = 0,
                 perc_reduce_tree_boxes = 0,
                 
                 road_width_mt = 5,
                 ext_mt_build_box = 0,
                 
                 ESAM_root = '/home/vaschetti/maxarSrc/models/EfficientSAM',
                 ESAM_num_parall_queries = 5,
                 smooth_patch_overlap = False,
                 use_separate_detect_config = False):
        
        #General
        self.batch_size = batch_size
        self.size = size
        self.stride = stride # Overlap between each patch = (size - stride)
        self.device = device
        self.smooth_patch_overlap = smooth_patch_overlap
        
        if not use_separate_detect_config: #if you are not using a separate detect_config then define here all the detection configuration
            #Grounding Dino (Trees)
            self.GD_root = Path(GD_root)
            self.CONFIG_PATH = self.GD_root / GD_config_file
            self.WEIGHTS_PATH = self.GD_root / GD_weights

            self.GD_model = GD_load_model(self.CONFIG_PATH, self.WEIGHTS_PATH).to(self.device)
            self.TEXT_PROMPT = TEXT_PROMPT
            self.BOX_THRESHOLD = BOX_THRESHOLD
            self.TEXT_THRESHOLD = TEXT_THRESHOLD
            self.max_area_GD_boxes_mt2 = max_area_GD_boxes_mt2
            self.min_ratio_GD_boxes_edges = min_ratio_GD_boxes_edges
            self.perc_reduce_tree_boxes = perc_reduce_tree_boxes

        #Efficient SAM
        self.efficient_sam = build_efficient_sam_vitt(os.path.join(ESAM_root, 'weights/efficient_sam_vitt.pt')).to(self.device)
        self.ESAM_num_parall_queries = ESAM_num_parall_queries
        
        #Roads
        self.road_width_mt = road_width_mt
        
        #Buildings
        self.ext_mt_build_box = ext_mt_build_box
        
        if not use_separate_detect_config:
            print('\n- GD model device:', next(self.GD_model.parameters()).device)
        print('- Efficient SAM device:', next(self.efficient_sam.parameters()).device)
    
    def __str__(self) -> str:
        return f'{self.TEXT_PROMPT = }\n{self.BOX_THRESHOLD = }\n{self.TEXT_THRESHOLD = }\n{self.max_area_GD_boxes_mt2 = }\n{self.min_ratio_GD_boxes_edges = }\n{self.perc_reduce_tree_boxes = }\n{self.road_width_mt = }\n{self.ext_mt_build_box = }'
    

class DetectConfig:

    def __init__(self,
                 batch_size = 1,
                 size = 600,
                 stride = 400,
                 
                 device = 'cuda',
                 GD_root = "/home/vaschetti/maxarSrc/models/GDINO",
                 GD_config_file = "GroundingDINO_SwinT_OGC.py",
                 GD_weights = "groundingdino_swint_ogc.pth",
                 
                 TEXT_PROMPT = 'bush', #'green tree'
                 BOX_THRESHOLD = 0.15,
                 TEXT_THRESHOLD = 0.30,
                 
                 max_area_GD_boxes_mt2 = 6000,
                 min_ratio_GD_boxes_edges = 0,
                 perc_reduce_tree_boxes = 0):
        
        #General
        self.batch_size = batch_size
        self.size = size
        self.stride = stride # Overlap between each patch = (size - stride)
        self.device = device
        
        #Grounding Dino (Trees)
        self.GD_root = Path(GD_root)
        self.CONFIG_PATH = self.GD_root / GD_config_file
        self.WEIGHTS_PATH = self.GD_root / GD_weights

        #self.GD_model = GD_load_model(self.CONFIG_PATH, self.WEIGHTS_PATH).to(self.device)
        self.TEXT_PROMPT = TEXT_PROMPT
        self.BOX_THRESHOLD = BOX_THRESHOLD
        self.TEXT_THRESHOLD = TEXT_THRESHOLD
        self.max_area_GD_boxes_mt2 = max_area_GD_boxes_mt2
        self.min_ratio_GD_boxes_edges = min_ratio_GD_boxes_edges
        self.perc_reduce_tree_boxes = perc_reduce_tree_boxes
        
        
    def __str__(self) -> str:
        return f'{self.TEXT_PROMPT = }\n{self.BOX_THRESHOLD = }\n{self.TEXT_THRESHOLD = }\n{self.max_area_GD_boxes_mt2 = }\n{self.min_ratio_GD_boxes_edges = }\n{self.perc_reduce_tree_boxes = }'
    
    
    