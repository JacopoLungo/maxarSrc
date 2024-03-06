from my_functions.detect import detect_utils
from groundingdino.util.inference import predict as GD_predict
import numpy as np
from typing import List
import geopandas as gpd
from my_functions.samplers import samplers_utils


def get_GD_boxes(img_batch: np.array, #b,h,w,c
                    GDINO_model,
                    TEXT_PROMPT,
                    BOX_THRESHOLD,
                    TEXT_THRESHOLD,
                    dataset_res,
                    device,
                    max_area_mt2 = 3000):
    
    batch_tree_boxes4Sam = []
    sample_size = img_batch.shape[1]
    num_trees4img = []

    for img in img_batch:
        image_transformed = detect_utils.GD_img_load(img)
        tree_boxes, logits, phrases = GD_predict(GDINO_model, image_transformed, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, device = device)
        num_trees4img.append(len(tree_boxes))
        tree_boxes4Sam = []
        if len(tree_boxes) != 0:
            keep_ix_tree_boxes = detect_utils.filter_on_box_area_mt2(tree_boxes, sample_size, dataset_res, max_area_mt2 = max_area_mt2)
            tree_boxes4Sam = detect_utils.GDboxes2SamBoxes(tree_boxes[keep_ix_tree_boxes], sample_size)
            batch_tree_boxes4Sam.append(tree_boxes4Sam)
    return batch_tree_boxes4Sam, np.array(num_trees4img)

def get_batch_buildings_boxes(batch_bbox: List, proj_buildings_gdf: gpd.GeoDataFrame, dataset_res, ext_mt = 10):
    batch_building_boxes = []
    num_build4img = []
    for bbox in batch_bbox:
        query_bbox_poly = samplers_utils.boundingBox_2_Polygon(bbox)
        index_MS_buildings = proj_buildings_gdf.sindex
        buildig_hits = index_MS_buildings.query(query_bbox_poly)
        num_build4img.append(len(buildig_hits))
        building_boxes = [] #append empty list if no buildings
        if len(buildig_hits) > 0:
            building_boxes = samplers_utils.rel_bbox_coords(proj_buildings_gdf.iloc[buildig_hits], query_bbox_poly.bounds, dataset_res, ext_mt=ext_mt)

        batch_building_boxes.append(np.array(building_boxes))

    return batch_building_boxes, np.array(num_build4img)