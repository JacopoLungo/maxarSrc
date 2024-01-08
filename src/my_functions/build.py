import glob
from pathlib import Path
import sys
import pyproj
import geopandas as gpd
from pyquadkey2 import quadkey
import pandas as pd
from shapely import geometry
import json
from typing import List, Tuple, Union
import time
import os
from torchgeo.datasets import stack_samples
from my_functions import samplers, geoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from my_functions import segment
import torch
import rasterio
from rasterio.features import rasterize
from my_functions.samplers_utils import path_2_tilePolygon


sys.path.append('/home/vaschetti/maxarSrc/models/EfficientSAM')
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

def get_region_name(event_name, metadata_root = '/home/vaschetti/maxarSrc/metadata'):
    metadata_root = Path(metadata_root)
    df = pd.read_csv(metadata_root / 'evet_id2State2Region.csv')
    return df[df['event_id'] == event_name]['region'].values[0]

def get_all_events(data_root = '/mnt/data2/vaschetti_data/maxar'):#TODO: restituire solo le cartelle che contengono un tiff
    """
    Get all the events in the data_root folder.
    Input:
        data_root: Example: '/mnt/data2/vaschetti_data/maxar'
    Output:
        all_events: List of events.
    """
    data_root = Path(data_root)
    all_events = []
    for event_name in glob.glob('*', root_dir=data_root):
        all_events.append(event_name)
    return all_events

def get_mosaics_names(event_name, data_root = '/mnt/data2/vaschetti_data/maxar', when = None):
    """
    Get all the mosaic names for an event.
    Input:
        event_name: Example: 'Gambia-flooding-8-11-2022'
        data_root: Example: '/mnt/data2/vaschetti_data/maxar'
        when: 'pre' or 'post'. Default matches both
    Output:
        all_mosaics: List of mosaic names. Example: ['104001007A565700', '104001007A565800']
    """
    data_root = Path(data_root)
    all_mosaics = []
    if when is not None:
        for mosaic_name in glob.glob('*', root_dir=data_root/event_name/when):
            all_mosaics.append(mosaic_name)
    else:
        for mosaic_name in glob.glob('**/*', root_dir=data_root/event_name):
            #all_mosaics.append(mosaic_name.split('/')[1])
            all_mosaics.append(os.path.split(mosaic_name)[1])
    return all_mosaics

def get_mosaic_bbox(event_name, mosaic_name, path_mosaic_metatada = '/home/vaschetti/maxarSrc/metadata/from_github_maxar_metadata/datasets', extra_mt = 0, return_proj_coords = False):
    """
    Get the bbox of a mosaic. It return the coordinates of the bottom left and top right corners.
    Input:
        event_name: Example: 'Gambia-flooding-8-11-2022'
        mosaic_name: It could be an element of the output of get_mosaics_names(). Example: '104001007A565700'
        path_mosaic_metatada: Path to the folder containing the geojson
        extra_mt: Extra meters added to all bbox sides. The center of the bbox remanis the same. (To be sure all elements are included)
        return_proj_coords: If True, it returns the coordinates in the projection of the mosaic.
    Output:
        pair of cordinates in format (lon, lat) or (x, y) if return_proj_coords is True
    """
    path_mosaic_metatada = Path(path_mosaic_metatada)
    file_name = mosaic_name + '.geojson'
    geojson_path = path_mosaic_metatada / event_name / file_name
    gdf = gpd.read_file(geojson_path)

    minx = sys.maxsize
    miny = sys.maxsize
    maxx = 0
    maxy = 0

    for _, row in gdf.iterrows():
        tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = [float(el) for el in row['proj:bbox'].split(',')]
        if tmp_minx < minx:
            minx = tmp_minx
        if tmp_miny < miny:
            miny = tmp_miny
        if tmp_maxx > maxx:
            maxx = tmp_maxx
        if tmp_maxy > maxy:
            maxy = tmp_maxy

    #enlarge bbox
    minx -= (extra_mt/2)
    miny -= (extra_mt/2)
    maxx += (extra_mt/2)
    maxy += (extra_mt/2)
    if not return_proj_coords:
        source_crs = gdf['proj:epsg'].values[0]
        target_crs = pyproj.CRS('EPSG:4326')
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        bott_left_lat, bott_left_lon = transformer.transform(minx, miny)
        top_right_lat, top_right_lon = transformer.transform(maxx, maxy)
        
        return ((bott_left_lon, bott_left_lat), (top_right_lon, top_right_lat)), gdf['proj:epsg'].values[0]
    
    return ((minx, miny), (maxx, maxy)), gdf['proj:epsg'].values[0]

def get_event_bbox(event_name, extra_mt = 0, when = None, return_proj_coords = False):
    
    minx = sys.maxsize
    miny = sys.maxsize
    maxx = 0
    maxy = 0

    crs_set = set()
    first_crs = None
    for mosaic_name in get_mosaics_names(event_name, when = when):
        ((tmp_minx, tmp_miny), (tmp_maxx, tmp_maxy)), crs = get_mosaic_bbox(event_name, mosaic_name, extra_mt = extra_mt, return_proj_coords = True)
        first_crs = crs if first_crs is None else first_crs
        transformer = pyproj.Transformer.from_crs(crs, first_crs)
        tmp_minx, tmp_miny = transformer.transform(tmp_minx, tmp_miny)
        tmp_maxx, tmp_maxy = transformer.transform(tmp_maxx, tmp_maxy)

        crs_set.add(crs)
        if tmp_minx < minx:
            minx = tmp_minx
        if tmp_miny < miny:
            miny = tmp_miny
        if tmp_maxx > maxx:
            maxx = tmp_maxx
        if tmp_maxy > maxy:
            maxy = tmp_maxy
    
    if not return_proj_coords:
        
        source_crs = first_crs #list(crs_set)[0]
        target_crs = pyproj.CRS('EPSG:4326')
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        bott_left_lat, bott_left_lon = transformer.transform(minx, miny)
        top_right_lat, top_right_lon = transformer.transform(maxx, maxy)

        return (bott_left_lon, bott_left_lat), (top_right_lon, top_right_lat)
    
    return (minx, miny), (maxx, maxy)

def intersecting_qks(bott_left_lon_lat: tuple, top_right_lon_lat: tuple, min_level=7, max_level=9):
    """
    Get the quadkeys that intersect the given bbox.
    Input:
        bott_left_lon_lat: Tuple with the coordinates of the bottom left corner. Example: (-16.5, 13.5)
        top_right_lon_lat: Tuple with the coordinates of the top right corner. Example: (-15.5, 14.5)
        min_level: Minimum level of the quadkeys
        max_level: Maximum level of the quadkeys
    """
    bott_left_lon, bott_left_lat = bott_left_lon_lat
    top_right_lon, top_right_lat = top_right_lon_lat
    qk_bott_left = quadkey.from_geo((bott_left_lat, bott_left_lon), level=max_level) #lat, lon
    qk_top_right = quadkey.from_geo((top_right_lat, top_right_lon), level=max_level) #lat, lon
    hits = qk_bott_left.difference(qk_top_right)
    candidate_hits = set()

    for hit in hits:
        current_qk = hit
        for _ in range(min_level, max_level):
            current_qk = current_qk.parent()
            candidate_hits.add(current_qk)
    hits.extend(candidate_hits)
    return [int(str(hit)) for hit in hits]

def qk_building_gdf(qk_list, csv_path = 'metadata/buildings_dataset_links.csv', dataset_crs = None, quiet = False):
    """
    Returns a geodataframe with the buildings of the country passed as input.
    It downloads the dataset from a link in the dataset-links.csv file.
    Coordinates are converted in the crs passed as input.

    Inputs:
        qk_list: the list of quadkeys to look for in the csv
        root: the root directory of the dataset-links.csv file
        dataset_crs: the crs in which to convert the coordinates of the buildings
        quiet: if True, it doesn't print anything
    """
    building_links_df = pd.read_csv(csv_path)
    country_links = building_links_df[building_links_df['QuadKey'].isin(qk_list)]

    if not quiet:
        print(f"Found {len(country_links)} links matching: {qk_list}")
    if len(country_links) == 0:
        print("No buildings for this region")
        return gpd.GeoDataFrame()
    gdfs = []
    for _, row in country_links.iterrows():
        df = pd.read_json(row.Url, lines=True)
        df["geometry"] = df["geometry"].apply(geometry.shape)
        gdf_down = gpd.GeoDataFrame(df, crs=4326)
        gdfs.append(gdf_down)

    gdfs = pd.concat(gdfs)
    if dataset_crs is not None: #se inserito il crs del dataset, lo converto
        gdfs = gdfs.to_crs(dataset_crs)
    return gdfs

def get_region_road_gdf(region_name, roads_root = '/mnt/data2/vaschetti_data/MS_roads'):
    """
    Get a gdf containing the roads of a region.
    Input:
        region_name: Name of the region. Example: 'AfricaWest-Full'
        roads_root: Root directory of the roads datasets
    """
    if region_name[-4:] != '.tsv':
        region_name = region_name + '.tsv'
    
    def custom_json_loads(s):
        try:
            return geometry.shape(json.loads(s)['geometry'])
        except:
            return geometry.LineString()

    roads_root = Path(roads_root)
    if region_name != 'USA.tsv':
        print('not USA', region_name)
        region_road_df = pd.read_csv(roads_root/region_name, names =['country', 'geometry'], sep='\t')
    else:
        print('is USA', region_name)
        region_road_df = pd.read_csv(roads_root/region_name, names =['geometry'], sep='\t')
    #region_road_df['geometry'] = region_road_df['geometry'].apply(json.loads).apply(lambda d: geometry.shape(d.get('geometry')))
    #slightly faster
    region_road_df['geometry'] = region_road_df['geometry'].apply(custom_json_loads)
    region_road_gdf = gpd.GeoDataFrame(region_road_df, crs=4326)
    return region_road_gdf

def filter_gdf_w_bbox(gbl_gdf: gpd.GeoDataFrame, bbox: Union[List[Tuple], Tuple[Tuple]]) -> gpd.GeoDataFrame:
    """
    Filter a geodataframe with a bbox.
    Input:
        gbl_gdf: the geodataframe to be filtered
        mosaic_bbox: Bounding box of the mosaic in format (lon, lat). Example: ((-16.5, 13.5), (-15.5, 14.5))
    Output:
        a filtered geodataframe
    
    """
    (minx, miny), (maxx, maxy) = bbox
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)] #lon lat
    query_bbox_poly = geometry.Polygon(vertices)
    
    hits = gbl_gdf.geometry.intersects(query_bbox_poly)
    #TODO: controllare magari funziona anche...
    #hits = gbl_gdf.sindex.query(query_bbox_poly)
    
    return gbl_gdf[hits]

def old_get_bbox_roads(mosaic_bbox: Union[List[Tuple], Tuple[Tuple]], region_name, roads_root = '/mnt/data2/vaschetti_data/MS_roads'):
    """
    Get a gdf containing the roads that intersect the mosaic_bbox.
    Input:
        mosaic_bbox: Bounding box of the mosaic in format (lon, lat). Example: ((-16.5, 13.5), (-15.5, 14.5))
        region_name: Name of the region. Example: 'AfricaWest-Full'
        roads_root: Root directory of the roads datasets
    """
    if region_name[-4:] != '.tsv':
        region_name = region_name + '.tsv'

    roads_root = Path(roads_root)
    road_df = pd.read_csv(roads_root/region_name, names =['country', 'geometry'], sep='\t')
    road_df['geometry'] = road_df['geometry'].apply(json.loads).apply(lambda d: geometry.shape(d.get('geometry')))
    road_gdf = gpd.GeoDataFrame(road_df, crs=4326)
    
    (minx, miny), (maxx, maxy) = mosaic_bbox
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)] #lon lat
    query_bbox_poly = geometry.Polygon(vertices)
    
    hits = road_gdf.geometry.intersects(query_bbox_poly)
    
    return road_gdf[hits]

"""unused
def get_boxes4FSam(tree_boxes_b, building_boxes_b, max_detect: int):
    boxes4FSam = [] 
    pad_value = -10
    for tree_detec, build_detec in zip(tree_boxes_b, building_boxes_b):
        tree_build_detect = np.concatenate((tree_detec, build_detec))
        pad_len = max_detect - (tree_build_detect.shape[0] + 1)
        pad_width = ((0,pad_len),(0, 0))
        boxes4FSam.append(np.pad(tree_build_detect, pad_width, constant_values=pad_value))
    
    return np.array(boxes4FSam)"""

def get_input_pts_and_lbs(tree_boxes_b: List, #list of array of shape (query_img_x, 4)
                          building_boxes_b: List, 
                          max_detect: int):
    input_lbs = []
    input_pts = []
    pts_pad_value = -10
    lbs_pad_value = 0
    for tree_detec, build_detec in zip(tree_boxes_b, building_boxes_b):
        tree_build_detect = np.concatenate((tree_detec, build_detec)) #(query_img_x, 4)
        lbs = np.array([[2,3]]*tree_build_detect.shape[0]) #(query_img_x, 2)

        pad_len = max_detect - (tree_build_detect.shape[0] + 1)
        print(pad_len)
        pad_width = ((0,pad_len),(0, 0))
        padded_tree_build_detect = np.pad(tree_build_detect, pad_width, constant_values=pts_pad_value)
        img_input_pts = np.expand_dims(padded_tree_build_detect, axis = 0).reshape(-1,2,2) # (max_queries, 2, 2)
        input_pts.append(img_input_pts)

        padded_lbs = np.pad(lbs, pad_width, constant_values = lbs_pad_value)# (max_queries, 2)
        input_lbs.append(padded_lbs)

    return np.array(input_pts), np.array(input_lbs) # (batch_size, max_queries, 2, 2), (batch_size, max_queries, 2)



from groundingdino.util.inference import load_model as GD_load_model
class SegmentConfig:
    """
    Config class for the segmentation pipeline.
    """
    def __init__(self,
                 batch_size,
                 size = 600,
                 stride = 300,
                 device = 'cuda',
                 GD_root = "/home/vaschetti/maxarSrc/models/GDINO",
                 GD_config_file = "GroundingDINO_SwinT_OGC.py",
                 GD_weights = "groundingdino_swint_ogc.pth",
                 TEXT_PROMPT = 'green tree',
                 BOX_TRESHOLD = 0.15,
                 TEXT_TRESHOLD = 0.30,
                 max_area_GD_boxes_mt2 = 6000,
                 ESAM_root = '/home/vaschetti/maxarSrc/models/EfficientSAM'):
        
        #General
        self.batch_size = batch_size
        self.size = size
        self.stride = stride
        self.device = device

        #Grounding Dino (Trees)
        self.GD_root = Path(GD_root)
        self.CONFIG_PATH = self.GD_root / GD_config_file
        self.WEIGHTS_PATH = self.GD_root / GD_weights

        self.GD_model = GD_load_model(self.CONFIG_PATH, self.WEIGHTS_PATH, device = self.device).to(self.device)
        print('- GD model device:', next(self.GD_model.parameters()).device)
        self.TEXT_PROMPT = TEXT_PROMPT
        self.BOX_TRESHOLD = BOX_TRESHOLD
        self.TEXT_TRESHOLD = TEXT_TRESHOLD
        self.max_area_GD_boxes_mt2 = max_area_GD_boxes_mt2

        #Efficient SAM
        self.efficient_sam = build_efficient_sam_vitt(os.path.join(ESAM_root, 'weights/efficient_sam_vitt.pt')).to(self.device)
        print('- Efficient SAM device:', next(self.efficient_sam.parameters()).device)
        

        #Roads



class Mosaic:
    def __init__(self,
                 name,
                 event
                 ):
        
        #Mosaic
        self.name = name
        self.event = event
        self.bbox, self.crs = get_mosaic_bbox(self.event.name,
                                          self.name,
                                          self.event.maxar_metadata_path,
                                          extra_mt=1000)
        
        self.when = list((self.event.maxar_root / self.event.name).glob('**/*'+self.name))[0].parts[-2]
        self.tiles_paths = list((self.event.maxar_root / self.event.name / self.when / self.name).glob('*.tif'))
        self.tiles_num = len(self.tiles_paths)

        #Roads
        self.road_gdf = None
        self.proj_road_gdf = None
        self.road_num = None

        #Buildings
        self.build_gdf = None
        self.proj_build_gdf = None
        self.build_num = None

    def set_road_gdf(self):
        if self.event.road_gdf is None:
            self.event.set_road_gdf()

        self.road_gdf = filter_gdf_w_bbox(self.event.road_gdf, self.bbox)
        self.proj_road_gdf =  self.road_gdf.to_crs(self.crs)
        self.road_num = len(self.road_gdf)
    
    def set_build_gdf(self):
        qk_hits = intersecting_qks(*self.bbox)
        self.build_gdf = qk_building_gdf(qk_hits, csv_path = self.event.buildings_ds_links_path)
        self.proj_build_gdf =  self.build_gdf.to_crs(self.crs)
        self.build_num = len(self.build_gdf)
    
    def __str__(self) -> str:
        return self.name
    
    def get_tile_road_mask_np(self, tile_path): 
        with rasterio.open(tile_path) as src:
            transform = src.transform
            tile_h = src.height
            tile_w = src.width
            out_meta = src.meta.copy()
        query_bbox_poly = path_2_tilePolygon(tile_path)
        road_lines = self.proj_roads_gdf[self.proj_roads_gdf.geometry.intersects(query_bbox_poly)]

        if len(road_lines) != 0:
            buffered_lines = road_lines.geometry.buffer(ext_mt)
            road_mask = rasterize(buffered_lines, out_shape=(tile_h, tile_w), transform=transform)
        else:
            print('No roads')
            road_mask = np.zeros((tile_h, tile_w))
        return road_mask
            
    def segment_tile(self, tile_path):
        seg_config = self.event.seg_config

        dataset = geoDatasets.Maxar(str(tile_path))
        sampler = samplers.MyBatchGridGeoSampler(dataset, batch_size=seg_config.batch_size, size=seg_config.size, stride=seg_config.stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)

        canvas = np.zeros((seg_config.size, seg_config.size, 3), dtype=np.uint8)

        for batch in tqdm(dataloader):          
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8')
            tree_boxes_b = segment.get_GD_boxes(img_b,
                                                seg_config.GD_model,
                                                seg_config.TEXT_PROMPT,
                                                seg_config.BOX_TRESHOLD,
                                                seg_config.TEXT_TRESHOLD,
                                                dataset.res,
                                                max_area_mt2 = seg_config.max_area_GD_boxes_mt2)
            
            building_boxes_b = segment.get_batch_buildings_boxes(batch['bbox'],
                                                                proj_buildings_gdf = self.proj_build_gdf,
                                                                dataset_res = dataset.res,
                                                                ext_mt = 10)
            tree_and_building_mask_b = None
            road_mask_b = None
            all_mask_b = None


            #fig, axs = plt.subplots(1, batch_size, figsize=(30, 30))
            #for i in range(batch_size):
            #    axs[i].imshow(img_b[i])
            #print(img_b.shape)
        
        #TODO: salvare la canvas come tiff
    
    """def segment_tile(self, tile_path, batch_size, seg_model, detect_model, size = 600, stride = 300):
        dataset = geoDatasets.Maxar(str(tile_path))
        sampler = samplers.MyBatchGridGeoSampler(dataset, batch_size=batch_size, size=size, stride=stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)

        canvas = np.zeros((dataset.height, dataset.width, 3), dtype=np.uint8)

        for batch in tqdm(dataloader):          
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8')
            tree_boxes_b = segment.get_GD_boxes(img_b, GDINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, dataset.res, max_area_mt2 =3000)
            building_boxes_b = None
            tree_and_building_mask_b = None
            road_mask_b = None


            #fig, axs = plt.subplots(1, batch_size, figsize=(30, 30))
            #for i in range(batch_size):
            #    axs[i].imshow(img_b[i])
            #print(img_b.shape)
        
        #TODO: salvare la canvas come tiff"""

    def segment_all_tiles(self):
        for tile_path in self.tiles_paths:
            self.segment_tile(tile_path)


class Event:
    def __init__(self,
                 name,
                 seg_config: SegmentConfig,
                 when = 'pre', #'pre', 'post' or None
                 maxar_root = '/mnt/data2/vaschetti_data/maxar',
                 maxar_metadata_path = '/home/vaschetti/maxarSrc/metadata/from_github_maxar_metadata/datasets',
                 region = 'infer'
                 ):
        #Segmentation
        self.seg_config = seg_config

        #Paths
        self.maxar_root = Path(maxar_root)
        self.buildings_ds_links_path = Path('/home/vaschetti/maxarSrc/metadata/buildings_dataset_links.csv')
        self.maxar_metadata_path = Path(maxar_metadata_path)
        
        #Event
        self.name = name
        self.when = when
        self.region_name = get_region_name(self.name) if region == 'infer' else region
        self.bbox = get_event_bbox(self.name, extra_mt=1000) #TODO può essere ottimizzata sfruttando i mosaici
        self.all_mosaics_names = get_mosaics_names(self.name, self.maxar_root, self.when)
    
        #Roads
        self.road_gdf = None

        #Mosaics
        self.mosaics = {}

        #Init mosaics
        for m_name in self.all_mosaics_names:
            self.mosaics[m_name] = Mosaic(m_name, self)


    #Roads methods
    def set_road_gdf(self): #set road_gdf for the event
        region_road_gdf = get_region_road_gdf(self.region_name)
        self.road_gdf = filter_gdf_w_bbox(region_road_gdf, self.bbox)

    def set_mos_road_gdf(self, mosaic_name): #set road_gdf for the mosaic
        if self.road_gdf is None:
            self.set_road_gdf()

        self.mosaics[mosaic_name].set_road_gdf()

    def set_all_mos_road_gdf(self): #set road_gdf for all the mosaics
        for mosaic_name, mosaic in self.mosaics.items():
            if mosaic.road_gdf is None:
                self.set_mos_road_gdf(mosaic_name)
    
    #Buildings methods
    def set_build_gdf_in_mos(self, mosaic_name):
        self.mosaics[mosaic_name].set_build_gdf()

    def set_build_gdf_all_mos(self):
        for mosaic_name, mosaic in self.mosaics.items():
            if mosaic.build_gdf is None:
                self.set_build_gdf_in_mos(mosaic_name)

    def get_mosaic(self, mosaic_name):
        return self.mosaics[mosaic_name]



