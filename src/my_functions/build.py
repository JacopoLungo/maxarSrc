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

def get_region_name(event_name, metadata_root = '/home/vaschetti/maxarSrc/metadata'):
    metadata_root = Path(metadata_root)
    df = pd.read_csv(metadata_root / 'evet_id2State2Region.csv')
    return df[df['event_id'] == event_name]['region'].values[0]

def get_all_events(data_root = '/mnt/data2/vaschetti_data/maxar'):
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
            all_mosaics.append(mosaic_name.split('/')[1])
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
        
        return ((bott_left_lon, bott_left_lat), (top_right_lon, top_right_lat)), target_crs
    
    return ((minx, miny), (maxx, maxy)), gdf['proj:epsg'].values[0]

def get_event_bbox(event_name, extra_mt = 0, when = None, return_proj_coords = False):
    
    minx = sys.maxsize
    miny = sys.maxsize
    maxx = 0
    maxy = 0

    crs_set = set()

    for mosaic_name in get_mosaics_names(event_name, when = when):
        ((tmp_minx, tmp_miny), (tmp_maxx, tmp_maxy)), crs = get_mosaic_bbox(event_name, mosaic_name, extra_mt = extra_mt, return_proj_coords = True)
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
        if len(crs_set) > 1:
            raise Exception('Different crs in the same event')
        else:
            source_crs = list(crs_set)[0]
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

    roads_root = Path(roads_root)
    region_road_df = pd.read_csv(roads_root/region_name, names =['country', 'geometry'], sep='\t')
    region_road_df['geometry'] = region_road_df['geometry'].apply(json.loads).apply(lambda d: geometry.shape(d.get('geometry')))
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

class Mosaic:
    def __init__(self, name):
        self.name = name
        self.tiles_path = []

    def add_tile(self, tile):
        self.tiles.append(tile)

    def remove_tile(self, tile):
        self.tiles.remove(tile)

    def get_tiles(self):
        return self.tiles


class Event:
    def __init__(self, name):
        self.name = name
        self.mosaics_name = []

    def add_mosaic(self, mosaic):
        self.mosaics.append(mosaic)

    def remove_mosaic(self, mosaic):
        self.mosaics.remove(mosaic)

    def get_mosaics(self):
        return self.mosaics