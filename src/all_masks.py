import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import os
import sys
sys.path.append('/home/vaschetti/maxarSrc/datasets_and_samplers')
from myGeoDatasets import Maxar
from mySamplers import MyGridGeoSampler
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples, unbind_samples
from samplers_utils import boundingBox_2_Polygon
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import supervision as sv
import torch
sys.path.append('/home/vaschetti/maxarSrc/creating_labels/MSBuildings')
from my_functions.segment import building_gdf, rel_bbox_coords, rel_polyg_coord, segment_buildings
sys.path.append('/home/vaschetti/maxarSrc/creating_labels/MSRoads')
from road_seg_utils import rel_road_lines, segment_roads, line2points, get_offset_lines, clear_roads, plotPoints, rmv_pts_out_img
import json

def segment_event(event_id, building_state_name, road_region):
    gdfs = building_gdf(building_state_name, dataset_crs)