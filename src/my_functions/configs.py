
from pathlib import Path
import os
import sys

from groundingdino.util.inference import load_model as GD_load_model
sys.path.append('/home/vaschetti/maxarSrc/models/EfficientSAM')
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt


class SegmentConfig:
    """
    Config class for the segmentation pipeline.
    """
    def __init__(self,
                 batch_size,
                 size = 600,
                 stride = 400,
                 device = 'cuda',
                 GD_root = "/home/vaschetti/maxarSrc/models/GDINO",
                 GD_config_file = "GroundingDINO_SwinT_OGC.py",
                 GD_weights = "groundingdino_swint_ogc.pth",
                 TEXT_PROMPT = 'green tree',
                 BOX_THRESHOLD = 0.15,
                 TEXT_THRESHOLD = 0.30,
                 max_area_GD_boxes_mt2 = 6000,
                 ESAM_root = '/home/vaschetti/maxarSrc/models/EfficientSAM',
                 ESAM_num_parall_queries = 5,
                 road_width_mt = 5,
                 smooth_patch_overlap = False):
        
        #General
        self.batch_size = batch_size
        self.size = size
        self.stride = stride # Overlap between each patch = (size - stride)
        self.device = device
        self.smooth_patch_overlap = smooth_patch_overlap
        
        #Grounding Dino (Trees)
        self.GD_root = Path(GD_root)
        self.CONFIG_PATH = self.GD_root / GD_config_file
        self.WEIGHTS_PATH = self.GD_root / GD_weights

        self.GD_model = GD_load_model(self.CONFIG_PATH, self.WEIGHTS_PATH).to(self.device)
        self.TEXT_PROMPT = TEXT_PROMPT
        self.BOX_THRESHOLD = BOX_THRESHOLD
        self.TEXT_THRESHOLD = TEXT_THRESHOLD
        self.max_area_GD_boxes_mt2 = max_area_GD_boxes_mt2

        #Efficient SAM
        self.efficient_sam = build_efficient_sam_vitt(os.path.join(ESAM_root, 'weights/efficient_sam_vitt.pt')).to(self.device)
        self.ESAM_num_parall_queries = ESAM_num_parall_queries
        
        #Roads
        self.road_width_mt = road_width_mt
        
        print('\n- GD model device:', next(self.GD_model.parameters()).device)
        print('- Efficient SAM device:', next(self.efficient_sam.parameters()).device)