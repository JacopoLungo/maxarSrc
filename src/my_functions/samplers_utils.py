import os
import json
import shapely
from typing import List
#import geopandas as gpd
#from my_functions import segment

def path_2_tilePolygon(tile_path, root = '/mnt/data2/vaschetti_data/maxar/metadata/from_github/datasets' ):
    """
    Create a shapely Polygon from a tile_path
    Example of a tile_path: '../Gambia-flooding-8-11-2022/pre/10300100CFC9A500/033133031213.tif'
    """
    event = tile_path.split('/')[-4]
    child = tile_path.split('/')[-2]
    tile = tile_path.split('/')[-1].replace(".tif", "")
    path_2_child_geojson = os.path.join(root, event, child +'.geojson')
    with open(path_2_child_geojson, 'r') as f:
        child_geojson = json.load(f)
    j = [el["properties"]["proj:geometry"] for el in child_geojson['features'] if el['properties']['quadkey'] == tile][0]
    tile_polyg = shapely.geometry.shape(j)
    return tile_polyg

def boundingBox_2_Polygon(bounding_box):
    """
    Create a shapely Polygon from a BoundingBox
    """
    minx, miny, maxx, maxy = bounding_box.minx, bounding_box.miny, bounding_box.maxx, bounding_box.maxy
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    bbox_polyg = shapely.geometry.Polygon(vertices)
    return bbox_polyg

def boundingBox_2_centralPoint(bounding_box):
    """
    Create a shapely Point from a BoundingBox
    """
    minx, miny, maxx, maxy = bounding_box.minx, bounding_box.miny, bounding_box.maxx, bounding_box.maxy
    return shapely.geometry.Point((minx + maxx)/2, (miny + maxy)/2)

"""def get_batch_buildings_boxes(batch_bbox: List, prj_buildings_gdf: gpd.GeoDataFrame, dataset_res, ext_mt = 10):
    batch_building_boxes = []
    for bbox in batch_bbox:
        query_bbox_poly = boundingBox_2_Polygon(bbox)
        index_MS_buildings = prj_buildings_gdf.sindex
        buildig_hits = index_MS_buildings.query(query_bbox_poly)
        building_boxes = [] #append empty list if no buildings
        if len(buildig_hits) > 0:
            building_boxes = segment.rel_bbox_coords(prj_buildings_gdf.iloc[buildig_hits], query_bbox_poly.bounds, dataset_res, ext_mt=ext_mt)

        batch_building_boxes.append(building_boxes)

    return batch_building_boxes"""