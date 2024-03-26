import argparse
import glob
import rasterio
from pathlib import Path
import os
import reverse_geocoder as rg
import pyproj
import geopandas as gpd
import sys

if Path.cwd().name != 'src':
    os.chdir('/home/vaschetti/maxarSrc/src')

from my_functions.assemble import names
from my_functions.assemble import holders

from my_functions.configs import SegmentConfig, DetectConfig

def main(): 

    events_names = names.get_all_events()
    
    parser = argparse.ArgumentParser(description='Segment Maxar Tiles')
    #event
    parser.add_argument('--event_ix', default = 6, type = int, help='Index of the event in the list events_names')
    parser.add_argument('--when', default = 'pre', choices=['pre', 'post', 'None'], help='Select the pre or post event mosaics')
    
    #Detect config
    parser.add_argument('--bs_det', default = 1, type = int, help = 'Batch size for the detection')
    parser.add_argument('--device_det', default = 'cuda:0', help='device to use for detection')
    
    parser.add_argument('--size_det', default = 600, type = int, help = 'Size of the patch for detection')
    parser.add_argument('--stride_det', default = 400, type = int, help = 'Stride of the patch for detection')
    
    parser.add_argument('--GD_root', default = "/home/vaschetti/maxarSrc/models/GDINO", help = 'Root of the grounding dino model')
    parser.add_argument('--GD_config_file', default = "GroundingDINO_SwinT_OGC.py", help = 'Config file of the grounding dino model')
    parser.add_argument('--GD_weights', default = "groundingdino_swint_ogc.pth", help = 'Weights of the grounding dino model')
    
    parser.add_argument('--text_prompt', default = 'green tree', help = 'Prompt for the grounding dino model')
    parser.add_argument('--box_threshold', default = 0.15, type = float, help = 'Threshold for the grounding dino model')
    parser.add_argument('--text_threshold', default = 0.30, type = float, help = 'Threshold for the grounding dino model')
    
    parser.add_argument('--max_area_GD_boxes_mt2', default = 6000, type = int, help = 'Max area of the boxes for the grounding dino model')
    parser.add_argument('--min_ratio_GD_boxes_edges', default = 0, type = float, help = 'Min ratio between edges of the tree boxes')
    parser.add_argument('--perc_reduce_tree_boxes', default = 0, type = float, help = 'Percentage of reduction of the tree boxes')
    
    #Segment config
    parser.add_argument('--bs_seg', default = 2, type = int, help = 'Batch size for the segmentation')
    parser.add_argument('--device_seg', default = 'cuda:0', help='device to use')
    
    parser.add_argument('--size_seg', default = 600, type = int, help = 'Size of the patch')
    parser.add_argument('--stride_seg', default = 400, type = int, help = 'Stride of the patch')
    
    parser.add_argument('--ext_mt_build_box', default = 0, type = int, help = 'Extra meter to enlarge building boxes')
    
    parser.add_argument('--road_width_mt', default = 5, type = int, help = 'Width of the road')    
    
    #Efficient SAM
    parser.add_argument('--ESAM_root', default = '/home/vaschetti/maxarSrc/models/EfficientSAM', help = 'Root of the efficient sam model')
    parser.add_argument('--ESAM_num_parall_queries', default = 5, type = int, help = 'Set the number of paraller queries to be processed')
    parser.add_argument('--out_dir_root', default = "/home/vaschetti/maxarSrc/output/tiff", help='output directory root')

    args = parser.parse_args()
        
    print("Selected Event: ", events_names[args.event_ix])
    
    
    det_config = DetectConfig(
                            batch_size = args.bs_det,
                            size = args.size_det,
                            stride = args.stride_det,
                            device = args.device_det,
                            GD_root = args.GD_root,
                            GD_config_file = args.GD_config_file,
                            GD_weights = args.GD_weights,
                            TEXT_PROMPT = args.text_prompt,
                            max_area_GD_boxes_mt2 = args.max_area_GD_boxes_mt2,
                            min_ratio_GD_boxes_edges = args.min_ratio_GD_boxes_edges,
                            perc_reduce_tree_boxes = args.perc_reduce_tree_boxes,
                            )
    
    seg_config = SegmentConfig(batch_size = args.bs_seg,
                           size = args.size_seg,
                           stride = args.stride_seg,
                           device = args.device_seg,
                           road_width_mt=args.road_width_mt,
                           ext_mt_build_box=args.ext_mt_build_box,
                           ESAM_root = args.ESAM_root,
                           ESAM_num_parall_queries = args.ESAM_num_parall_queries,
                           use_separate_detect_config=True
                           )
    
    event = holders.Event(events_names[args.event_ix],
                          seg_config = seg_config,
                          det_config = det_config,
                          when=args.when)
    
    all_mosaics_names = event.all_mosaics_names
    
    
    #event.seg_all_mosaics() #this segment all the mosiacs in the event
    
    m0 = event.mosaics[all_mosaics_names[0]]
    #m0.segment_all_tiles() #this segment all tiles in the mosaic
    
    m0_tile_17_path = m0.tiles_paths[17]
    m0.segment_tile(m0_tile_17_path, args.out_dir_root, glbl_det = True)


if __name__ == "__main__":
    main()