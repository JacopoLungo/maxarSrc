import os
import json
import shapely
from typing import List
from pathlib import Path
import numpy as np
import geopandas as gpd
from maxarseg.geo_datasets import geoDatasets
from shapely.geometry.polygon import Polygon
from typing import Optional

#from maxarseg import segment

def path_2_tile_aoi(tile_path, root = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/metadata/from_github/datasets' ):
    """
    Create a shapely Polygon from a tile_path
    Example of a tile_path: '../Gambia-flooding-8-11-2022/pre/10300100CFC9A500/033133031213.tif'
    """
    if isinstance(tile_path, str):
        event = tile_path.split('/')[-4]
        child = tile_path.split('/')[-2]
        tile = tile_path.split('/')[-1].replace(".tif", "")
    elif isinstance(tile_path, Path):
        event = tile_path.parts[-4]
        child = tile_path.parts[-2]
        tile = tile_path.parts[-1].replace(".tif", "")
    else:
        raise TypeError("tile_path must be a string or a Path object")
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

def xyxy_2_Polygon(xyxy_box):
    """
    Create a shapely Polygon from a xyxy box
    """
    if not len(xyxy_box) == 4: #allow for a tuple of 2 points. E.g. ((minx, miny), (maxx, maxy))
        minx, miny = xyxy_box[0]
        maxx, maxy = xyxy_box[1]
    else:    
        minx, miny, maxx, maxy = xyxy_box
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    return shapely.geometry.Polygon(vertices)

def xyxyBox2Polygon(xyxy_box):
    """
    Create a shapely Polygon from a xyxy box
    """
    minx, miny, maxx, maxy = xyxy_box
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    bbox_polyg = shapely.geometry.Polygon(vertices)
    return bbox_polyg
    

def boundingBox_2_centralPoint(bounding_box):
    """
    Create a shapely Point from a BoundingBox
    """
    minx, miny, maxx, maxy = bounding_box.minx, bounding_box.miny, bounding_box.maxx, bounding_box.maxy
    return shapely.geometry.Point((minx + maxx)/2, (miny + maxy)/2)

def align_bbox(bbox: Polygon):
    """
    Turn the polygon into a bbox axis aligned
    """
    minx, miny, maxx, maxy = bbox.bounds
    return minx, miny, maxx, maxy

def rel_bbox_coords(geodf:gpd.GeoDataFrame,
                    ref_coords:tuple,
                    res,
                    ext_mt: int = 0) -> List[List[float]]:
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
    ref_minx, ref_maxy = ref_coords[0], ref_coords[3] #coords of top left corner of the patch sample extracted from the tile
    #print('\nref_coords top left: ', ref_minx, ref_maxy )
    for geom in geodf['geometry']:
        minx, miny, maxx, maxy = align_bbox(geom)
        if ext_mt != None or ext_mt != 0:
            minx -= (ext_mt / 2)
            miny -= (ext_mt / 2)
            maxx += (ext_mt / 2)
            maxy += (ext_mt / 2)

        rel_bbox_coords = list(np.array([minx - ref_minx, ref_maxy - maxy, maxx - ref_minx, ref_maxy - miny]) / res)
        result.append(rel_bbox_coords)
    
    return result

def tile_sizes(dataset: geoDatasets.MxrSingleTile) -> tuple:
    """
    Returns the sizes of the tile given the path
    It uses the 
    """
    bounds = dataset.bounds
    x_size_pxl = (bounds.maxy - bounds.miny) / dataset.res
    y_size_pxl = (bounds.maxx - bounds.minx) / dataset.res
    
    if x_size_pxl % 1 != 0 or y_size_pxl % 1 != 0:
        raise ValueError("The sizes of the tile are not integers")
    
    return (int(x_size_pxl), int(y_size_pxl))