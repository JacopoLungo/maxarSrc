import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Polygon, LineString, MultiPoint, Point
import os
import sys
sys.path.append('/home/vaschetti/maxarSrc/datasets_and_samplers')
from my_functions.geoDatasets import Maxar
from my_functions.samplers import MyGridGeoSampler
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples, unbind_samples
from my_functions.samplers_utils import boundingBox_2_Polygon
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import supervision as sv
import torch

from rasterio.features import rasterize
from skimage import morphology

import groundingdino.datasets.transforms as T
from PIL import Image
from torchvision.ops import box_convert
from typing import Union, List

from torchvision import transforms
from groundingdino.util.inference import predict as GD_predict

#############
# Buildings
#############

def building_gdf(country, csv_path = '/home/vaschetti/maxarSrc/metadata/buildings_dataset_links.csv', dataset_crs = None, quiet = False):
    """
    Returns a geodataframe with the buildings of the country passed as input.
    It downloads the dataset from a link in the dataset-links.csv file.
    Coordinates are converted in the crs passed as input.
    Inputs:
        country: the country of which to download the buildings. Example: 'Tanzania'
        root: the root directory of the dataset-links.csv file
        dataset_crs: the crs in which to convert the coordinates of the buildings
        quiet: if True, it doesn't print anything
    """
    dataset_links = pd.read_csv(csv_path)
    country_links = dataset_links[dataset_links.Location == country]
    #TODO: eventualmente filtrare anche sul quadkey dell evento
    if not quiet:
        print(f"Found {len(country_links)} links for {country}")

    gdfs = []
    for _, row in country_links.iterrows():
        df = pd.read_json(row.Url, lines=True)
        df["geometry"] = df["geometry"].apply(shape)
        gdf_down = gpd.GeoDataFrame(df, crs=4326)
        gdfs.append(gdf_down)

    gdfs = pd.concat(gdfs)
    if dataset_crs is not None: #se inserito il crs del dataset, lo converto
        gdfs = gdfs.to_crs(dataset_crs)
    return gdfs

def rel_bbox_coords(geodf:gpd.GeoDataFrame,
                    ref_coords:tuple,
                    res,
                    ext_mt = None):
    """
    Returns the relative coordinates of a bbox w.r.t. a reference bbox in the 'geometry' column.
    Goes from absolute geo coords to relative coords in the image.

    Inputs:
        geodf: dataframe with bboxes
        ref_coords: a tuple in the format (minx, miny, maxx, maxy)
        res: resolution of the image
        ext_mt: meters to add to each edge of the box (the center remains fixed)
    Returns:
        a list of tuples with the relative coordinates of the bboxes [(minx, miny, maxx, maxy), ...]
    """
    result = []
    ref_minx, ref_maxy = ref_coords[0], ref_coords[3] #coords of top left corner of the square sample extracted from the tile
    #print('\nref_coords top left: ', ref_minx, ref_maxy )
    for geom in geodf['geometry']:
        building_minx, building_miny, building_maxx, building_maxy = geom.bounds
        if ext_mt != None:
            building_minx -= (ext_mt / 2)
            building_miny -= (ext_mt / 2)
            building_maxx += (ext_mt / 2)
            building_maxy += (ext_mt / 2)

        rel_bbox_coords = list(np.array([building_minx - ref_minx, ref_maxy - building_maxy, building_maxx - ref_minx, ref_maxy - building_miny]) / res)
        result.append(rel_bbox_coords)
    
    return result

def rel_polyg_coord(geodf:gpd.GeoDataFrame,
                    ref_coords:tuple,
                    res):
    """
    Returns the relative coordinates of a polygon w.r.t. a reference bbox.
    Goes from absolute geo coords to relative coords in the image.

    Inputs:
        geodf: dataframe with polygons in the 'geometry' column
        ref_coords: a tuple in the format (minx, miny, maxx, maxy)
        res: resolution of the image
    Returns:
        a list of lists of tuples with the relative coordinates of the bboxes [[(p1_minx1, p1_miny1), (p1_minx2, p1_miny2), ...], [(p2_minx1, p2_miny1), (p2_minx2, p2_miny2), ...], ...]
    """
    result = []
    ref_minx, ref_maxy = ref_coords[0], ref_coords[3] #coords of top left corner

    for geom in geodf['geometry']:
        x_s, y_s = geom.exterior.coords.xy
        rel_x_s = (np.array(x_s) - ref_minx) / res
        rel_y_s = (ref_maxy - np.array(y_s)) / res
        rel_coords = list(zip(rel_x_s, rel_y_s))
        result.append(rel_coords)
    return result

def get_batch_buildings_boxes(batch_bbox: List, proj_buildings_gdf: gpd.GeoDataFrame, dataset_res, ext_mt = 10):
    batch_building_boxes = []
    num_build4img = []
    for bbox in batch_bbox:
        query_bbox_poly = boundingBox_2_Polygon(bbox)
        index_MS_buildings = proj_buildings_gdf.sindex
        buildig_hits = index_MS_buildings.query(query_bbox_poly)
        num_build4img.append(len(buildig_hits))
        building_boxes = [] #append empty list if no buildings
        if len(buildig_hits) > 0:
            building_boxes = rel_bbox_coords(proj_buildings_gdf.iloc[buildig_hits], query_bbox_poly.bounds, dataset_res, ext_mt=ext_mt)

        batch_building_boxes.append(np.array(building_boxes))

    return batch_building_boxes, np.array(num_build4img)

def segment_buildings(predictor, building_boxes, img4Sam: np.array, use_bbox = True, use_center_points = False):
    """
    Segment the buildings in the image using the predictor passed as input.
    The image has to be encoded the image before calling this function.
    Inputs:
        predictor: the predictor to use for the segmentation
        building_boxes: a list of tuples containing the building's bounding boxes in formtat (minx, miny, maxx, maxy) = (top left corner, bottom right corner)
        img4Sam: the image previously encoded
        use_bbox: if True, the bounding boxes are used for the segmentation
        use_center_points: if True, the center points of the bounding boxes are used for the segmentation

    Returns:
        mask: a np array of shape (1, h, w). The mask is True where there is a building, False elsewhere
        bboxes: a list of tuples containing the bounding boxes of the buildings used for the segmentation
        #!used_points: a np array of shape (n, 2) where n is the number of buildings. The array contains the center points of the bounding boxes of the buildings in the image
    """

    building_boxes_t = torch.tensor(building_boxes, device=predictor.device)
    
    transformed_boxes = None
    if use_bbox:
        transformed_boxes = predictor.transform.apply_boxes_torch(building_boxes_t, img4Sam.shape[:2])
    
    transformed_points = None
    transformed_point_labels = None
    """if use_center_points: #TODO: aggiustare l'utilizzo di punti, al momento non funziona
        point_coords = torch.tensor([[(sublist[0] + sublist[2])/2, (sublist[1] + sublist[3])/2] for sublist in building_boxes_t], device=predictor.device)
        point_labels = torch.tensor([1] * point_coords.shape[0], device=predictor.device)[:, None]
        transformed_points = predictor.transform.apply_coords_torch(point_coords, img4Sam.shape[:2]).unsqueeze(1)
        transformed_point_labels = point_labels[:, None]"""

    masks, _, _ = predictor.predict_torch(
                point_coords=transformed_points,
                point_labels=transformed_point_labels,
                boxes=transformed_boxes,
                multimask_output=False,
            )
    #mask is a tensor (n, 1, h, w) where n = number of mask = numb. of input boxes
    mask = np.any(masks.cpu().numpy(), axis = 0)

    used_boxes = None
    if use_bbox:
        used_boxes = building_boxes

    used_points = None
    """if use_center_points:
        used_points = point_coords.cpu().numpy()"""

    return mask, used_boxes, used_points #returns all the np array


#############
# Roads with SAM
#############

def rel_road_lines(geodf: gpd.GeoDataFrame,
                    query_bbox_poly: Polygon,
                    res):
    """
    Given a Geodataframe containing Linestrings with geo coords, 
    returns the relative coordinates of those lines w.r.t. a reference bbox

    Inputs:
        geodf: GeoDataFrame containing the Linestring
        query_bbox_poly: Polygon of the reference bbox
        res: resolution of the image
    Returns:
        result: list of LineString with the relative coordinates
    """
    ref_coords = query_bbox_poly.bounds
    ref_minx, ref_maxy = ref_coords[0], ref_coords[3] #coords of top left corner

    result = []
    for line in geodf.geometry:
        x_s, y_s = line.coords.xy

        rel_x_s = (np.array(x_s) - ref_minx) / res
        rel_y_s = (ref_maxy - np.array(y_s)) / res
        rel_coords = list(zip(rel_x_s, rel_y_s))
        line = LineString(rel_coords)
        result.append(line)
    return result

def line2points(lines: Union[LineString, List[LineString]], points_dist) -> List[Point]:
    """
    Given a single or a list of shapely.LineString,
    returns a list of shapely points along all the lines, spaced by points_dist
    """
    if not isinstance(lines, list):
        lines = [lines]
    points = []
    for line in lines:
        points.extend([line.interpolate(dist) for dist in np.arange(0, line.length, points_dist)])
    return points

def get_offset_lines(lines: Union[LineString, List[LineString]], distance=35):
    """
    Create two offset lines from a single or list of shapely.LineString at distance = 'distance'
    """
    if not isinstance(lines, list):
        lines = [lines]
    
    offset_lines = []
    for line in lines:
        for side in [-1, +1]:
            offset_lines.append(line.offset_curve(side*distance ))
    return offset_lines

def clear_roads(lines: Union[LineString, List[LineString]], bg_points, distance) -> List[Point]:
    """
    Given a list of shapely.LineString and a list of shapely.Point,
    remove bg points that may be on the road
    """
    candidate_bg_pts = bg_points
    final_bg_pts = set(bg_points)

    if not isinstance(lines, list):
        lines = [lines]

    for line in lines:
        line_space = line.buffer(distance)
        for point in candidate_bg_pts:
            if line_space.contains(point):
                final_bg_pts.discard(point)
        
    return list(final_bg_pts)

def rmv_rnd_fraction(points, fraction_to_keep):
    """
    Removes a random fraction of the points
    """
    np.random.shuffle(points)
    points = points[:int(len(points)*fraction_to_keep)]
    return points

def rmv_pts_out_img(points: np.array, sample_size)-> np.array:
    """
    Given a np.array of points (n, 2),
    removes points outside the image
    """
    if len(points) != 0:
        points = points[np.logical_and(np.logical_and(points[:, 0] >= 0, points[:, 0] < sample_size), np.logical_and(points[:, 1] >= 0, points[:, 1] < sample_size))]
    return points

def segment_roads(predictor,
                  road_lines: Union[LineString, List[LineString]],
                  sample_size,
                  img4Sam = None,
                  road_point_dist = 50,
                  bg_point_dist = 80,
                  offset_distance = 50,
                  do_clean_mask = True):
    """
    Segment the roads in the image using the predictor passed as input.
    If passed as input the image is encoded on the fly, otherwise it has to be encoded before calling this function.

    Inputs:
        predictor: the predictor to use for the segmentation
        road_lines: a list of shapely.LineString containing the roads
        sample_size: the size of the image
        img4Sam: the image to encode if not already encoded
        road_point_dist: the distance between two points on the road
        bg_point_dist: the distance between two points in the road's offset lines
        offset_distance: the offset distance
        do_clean_mask: if True, the mask is cleaned by removing parts outside the offset lines

    Returns:
        final_mask: a np array of shape (1, h, w). The mask is True where there is a road, False elsewhere
        final_pt_coords4Sam: a np array of shape (n, 2) where n is the number of points. The array contains the coordinates of the points used for the segmentation
        final_labels4Sam: a np array of shape (n,) where n is the number of points. The array contains the labels of the points used for the segmentation
    """
    
    #Decide if encoding here or outside the function
    if img4Sam is not None:
        predictor.set_image(img4Sam)
    
    #initialize an empty mask
    final_mask = np.full((sample_size, sample_size), False)
    
    final_pt_coords4Sam = []
    final_labels4Sam = []

    if not isinstance(road_lines, list):
        road_lines = [road_lines]

    for road in road_lines:
        road_pts = line2points(road, road_point_dist) #turn the road into a list of shapely points
        np_roads_pts = np.array([list(pt.coords)[0] for pt in road_pts]) #turn the shapely points into a numpy array
        np_roads_pts = rmv_pts_out_img(np_roads_pts, sample_size) #remove road points outside the image
        np_road_labels = np.array([1]*np_roads_pts.shape[0]) #create the labels for the road points
        
        bg_lines = get_offset_lines(road, offset_distance) #create two offset lines from the road
        bg_pts = line2points(bg_lines, bg_point_dist) #turn the offset lines into a list of shapely points
        bg_pts = clear_roads(road_lines, bg_pts, offset_distance - 4) #remove bg points that may be on other roads
        np_bg_pts = np.array([list(pt.coords)[0] for pt in bg_pts]) #turn the shapely points into a numpy array
        np_bg_pts = rmv_pts_out_img(np_bg_pts, sample_size) #remove road points outside the image
        np_bg_labels = np.array([0]*np_bg_pts.shape[0]) #create the labels for the bg points

        if len(np_bg_labels) == 0 or len(np_road_labels) < 2: #if there are no bg_points or 0 or 1 road points skip the road
            continue

        pt_coords4Sam = np.concatenate((np_roads_pts, np_bg_pts)) #tmp list
        labels4Sam = np.concatenate((np_road_labels, np_bg_labels))

        final_pt_coords4Sam.extend(pt_coords4Sam.tolist()) #global list
        final_labels4Sam.extend(labels4Sam.tolist()) #global list

        mask, _, _ = predictor.predict(
                point_coords=pt_coords4Sam,
                point_labels=labels4Sam,
                multimask_output=False,
            )
        final_mask = np.logical_or(final_mask, mask[0])

    if do_clean_mask:
        final_mask = clean_mask(road_lines, final_mask, offset_distance - 10) #TODO: eventualmente aggiungere un parametro per l'additional_cleaning       
    
    return final_mask[np.newaxis, :], np.array(final_pt_coords4Sam), np.array(final_labels4Sam)
    
def clean_mask(road_lines: Union[LineString, List[LineString]],
               final_mask_2d: np.array,
               offset_distance,
               additional_cleaning = False):
    """
    Clean the mask by removing parts outside the offset lines.
    The additional_cleaning parameter is used to remove small holes and small objects from the mask.
    """

    if not isinstance(road_lines, list):
        road_lines = [road_lines]

    line_buffers = [line.buffer(offset_distance) for line in road_lines]
    buffer_roads = rasterize(line_buffers, out_shape=final_mask_2d.shape)
    clear_mask = np.logical_and(final_mask_2d, buffer_roads)
    
    if additional_cleaning: #TODO: controllare meglio cosa fanno queste funzioni e tunare i parametri
        clear_mask = morphology.remove_small_holes(clear_mask, area_threshold=500)
        clear_mask = morphology.remove_small_objects(clear_mask, min_size=500)
        clear_mask = morphology.binary_opening(clear_mask)
        clear_mask = morphology.binary_closing(clear_mask)
    
    return clear_mask

#############
# Roads with buffer
#############

# Method not used in the final version
def get_road_masks_b(query_bbox_b, proj_roads_gdf: gpd.GeoDataFrame, sample_size, dataset_res, ext_mt = 10):
    for bbox in query_bbox_b:
        query_bbox_poly = boundingBox_2_Polygon(bbox)
        road_hits = proj_roads_gdf.geometry.intersects(query_bbox_poly)
        if len(road_hits) != 0:
            queried_proj_roads_gdf = proj_roads_gdf[road_hits]
            road_lines = rel_road_lines(queried_proj_roads_gdf, query_bbox_poly, dataset_res)
            buffered_lines = road_lines.buffer(ext_mt)
            road_mask_b = rasterize(buffered_lines, out_shape=(1, sample_size, sample_size))
        
    

#############
#Trees
#############

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

def get_GD_boxes(img_batch: np.array, #b,h,w,c
                    GDINO_model,
                    TEXT_PROMPT,
                    BOX_TRESHOLD,
                    TEXT_TRESHOLD,
                    dataset_res,
                    device,
                    max_area_mt2 = 3000):
    
    batch_tree_boxes4Sam = []
    sample_size = img_batch.shape[1]
    num_trees4img = []

    for img in img_batch:
        image_transformed = GD_img_load(img)
        tree_boxes, logits, phrases = GD_predict(GDINO_model, image_transformed, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, device = device)
        num_trees4img.append(len(tree_boxes))
        tree_boxes4Sam = []
        if len(tree_boxes) != 0:
            keep_ix_tree_boxes = filter_on_box_area_mt2(tree_boxes, sample_size, dataset_res, max_area_mt2 = max_area_mt2)
            tree_boxes4Sam = GDboxes2SamBoxes(tree_boxes[keep_ix_tree_boxes], sample_size)
            batch_tree_boxes4Sam.append(tree_boxes4Sam)
    return batch_tree_boxes4Sam, np.array(num_trees4img)

#############
#General
#############

def segment_from_boxes(predictor, boxes, img4Sam, use_bbox = True, use_center_points = False):
    """
    Segment the buildings in the image using the predictor passed as input.
    The image has to be encoded the image before calling this function.
    Inputs:
        predictor: the predictor to use for the segmentation
        building_boxes: a list of tuples or a 2d np.array (b, 4) containing the building's bounding boxes. A single box is in the format (minx, miny, maxx, maxy) = (top left corner, bottom right corner)
        img4Sam: the image previously encoded
        use_bbox: if True, the bounding boxes are used for the segmentation
        use_center_points: if True, the center points of the bounding boxes are used for the segmentation

    Returns:
        mask: a np array of shape (1, h, w). The mask is True where there is a building, False elsewhere
        bboxes: a list of tuples containing the bounding boxes of the buildings used for the segmentation
        #!used_points: a np array of shape (n, 2) where n is the number of buildings. The array contains the center points of the bounding boxes of the buildings in the image
    """

    boxes_t = torch.tensor(boxes, device=predictor.device)
    
    #if use_bboxes
    transformed_boxes = None
    if use_bbox:
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_t, img4Sam.shape[:2])
    
    #if use points
    transformed_points = None
    transformed_point_labels = None
    if use_center_points: #TODO: aggiustare l'utilizzo di punti, al momeno non funziona
        point_coords = torch.tensor([[(sublist[0] + sublist[2])/2, (sublist[1] + sublist[3])/2] for sublist in building_boxes_t], device=predictor.device)
        point_labels = torch.tensor([1] * point_coords.shape[0], device=predictor.device)[:, None]
        transformed_points = predictor.transform.apply_coords_torch(point_coords, img4Sam.shape[:2]).unsqueeze(1)
        transformed_point_labels = point_labels[:, None]

    masks, _, _ = predictor.predict_torch(
                point_coords=transformed_points,
                point_labels=transformed_point_labels,
                boxes=transformed_boxes,
                multimask_output=False,
            )
    #mask è un tensore di dimensione (n, 1, h, w) dove n è il numero di maschere (=numero di box passate)
    mask = np.any(masks.cpu().numpy(), axis = 0)

    used_boxes = None
    if use_bbox:
        used_boxes = boxes

    used_points = None
    if use_center_points:
        used_points = point_coords.cpu().numpy()

    return mask, used_boxes, used_points #returna tutti np array


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

def discern(all_mask_b: np.array, num_trees4img:np.array, num_build4img: np.array):
    h, w = all_mask_b.shape[2:]
    tree_mask_b = build_mask_b = pad_mask_b = np.full((1, h, w), False)

    all_mask_b = np.greater_equal(all_mask_b, 0) #from logits to bool
    for all_mask, tree_ix, build_ix in zip (all_mask_b, num_trees4img, num_build4img):
        #all_mask.shape = (num_mask, h, w)
        tree_mask = all_mask[ : tree_ix].any(axis=0) #(h, w)
        tree_mask_b = np.concatenate((tree_mask_b, tree_mask[None, ...]), axis=0) #(b, h, w)

        build_mask = all_mask[tree_ix : (tree_ix + build_ix)].any(axis=0)
        build_mask_b = np.concatenate((build_mask_b, build_mask[None, ...]), axis=0)
        
        pad_mask = all_mask[(tree_ix + build_ix) : ].any(axis=0)
        pad_mask_b = np.concatenate((pad_mask_b, pad_mask[None, ...]), axis=0)
    
    return tree_mask_b[1:], build_mask_b[1:], pad_mask_b[1:] #all (b, h, w), slice out the first element

#Use the batch version of this function
def rmv_mask_overlap(overlapping_masks: np.array):
    """
    Remove overlapping between the masks. Giving priority according to the inverse of the order of
    the masks.
    Third (building) mask has priority over second (trees) mask, and so on.
    """
    disjoined_masks = np.copy(overlapping_masks)
    for i in range(overlapping_masks.shape[0] - 1):
        sum_mask = np.sum(overlapping_masks[i:], axis=0)
        disjoined_masks[i] = np.where(sum_mask > 1, False, overlapping_masks[i])

    return disjoined_masks

def rmv_mask_b_overlap(overlapping_masks_b: np.array): #(b, c, h, w)
    """
    Remove overlapping between the masks. Giving priority according to the inverse of the order of
    the masks.
    Third (building) mask has priority over second (trees) mask, and so on.
    """

    disjoined_masks_b = np.copy(overlapping_masks_b)
    for i in range(overlapping_masks_b.shape[1] - 1):
        sum_mask = np.sum(overlapping_masks_b[:,i:], axis=1)
        disjoined_masks_b[:,i] = np.where(sum_mask > 1, False, overlapping_masks_b[:, i])
    
    return disjoined_masks_b

