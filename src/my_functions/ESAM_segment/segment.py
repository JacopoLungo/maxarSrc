#import pandas as pd
#import geopandas as gpd
#from shapely.geometry import shape, Polygon, LineString, MultiPoint, Point
#import sys
#sys.path.append('/home/vaschetti/maxarSrc/datasets_and_samplers')
#from my_functions.build.geoDatasets import Maxar
#from my_functions.samplers import MyGridGeoSampler
#from torch.utils.data import DataLoader
#from torchgeo.datasets import stack_samples, unbind_samples
#from my_functions.samplers_utils import boundingBox_2_Polygon
#import matplotlib.patches as patches
#import matplotlib.pyplot as plt
#import cv2
#import supervision as sv
#import groundingdino.datasets.transforms as T
#from PIL import Image
#from torchvision.ops import box_convert
#from typing import Union, List
#from torchvision import transforms
#from rasterio.features import rasterize



import numpy as np
import torch


def ESAM_from_inputs(original_img_tsr: torch.tensor, #b, c, h, w
                    input_points: torch.tensor, #b, max_queries, 2, 2
                    input_labels: torch.tensor, #b, max_queries, 2
                    efficient_sam,
                    num_parall_queries: int = 50,
                    device = 'cpu',
                    empty_cuda_cache = True):
    
    img_b_tsr = original_img_tsr.div(255)
    batch_size, _, input_h, input_w = img_b_tsr.shape
    
    img_b_tsr = img_b_tsr.to(device)
    input_points = input_points.to(device)
    input_labels = input_labels.to(device)

    image_embeddings = efficient_sam.get_image_embeddings(img_b_tsr)
    
    stop = input_points.shape[1]
    for i in range(0, stop , num_parall_queries):
        start_idx = i
        end_idx = min(i + num_parall_queries, stop)
        predicted_logits, predicted_iou = efficient_sam.predict_masks(image_embeddings,
                                                                input_points[:, start_idx: end_idx],
                                                                input_labels[:, start_idx: end_idx],
                                                                multimask_output=True,
                                                                input_h = input_h,
                                                                input_w = input_w,
                                                                output_h=input_h,
                                                                output_w=input_w)
        
        if i == 0:
            print('predicetd_logits:', predicted_logits.shape)
            np_complete_masks = predicted_logits[:,:,0].cpu().detach().numpy()
        else:
            np_complete_masks = np.concatenate((np_complete_masks, predicted_logits[:,:,0].cpu().detach().numpy()), axis=1)
        if empty_cuda_cache:
            del predicted_logits, predicted_iou
            torch.cuda.empty_cache()
    
    return np_complete_masks