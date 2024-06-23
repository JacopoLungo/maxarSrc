from rasterio.features import rasterize
from pathlib import Path
import glob
import os
import json
import shapely
import rasterio

def path_2_tile_aoi(tile_path, root = './metadata/from_github_maxar_metadata/datasets' ):
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
    
    try:
        path_2_child_geojson = os.path.join(root, event, child +'.geojson')
        with open(path_2_child_geojson, 'r') as f:
            child_geojson = json.load(f)
    except:
        file_pattern = str(os.path.join(root, event, child + '*inv.geojson'))
        file_list = glob.glob(f"{file_pattern}")
        assert len(file_list) == 1, f"Found {len(file_list)} files with pattern {file_pattern}. Expected 1 file."
        path_2_child_geojson = file_list[0]
        with open(path_2_child_geojson, 'r') as f:
            child_geojson = json.load(f)
    
    
    j = [el["properties"]["proj:geometry"] for el in child_geojson['features'] if el['properties']['quadkey'] == tile][0]
    tile_polyg = shapely.geometry.shape(j)
    return tile_polyg

def path_to_aoi_mask(tile_path):
    with rasterio.open(tile_path) as src:
        transform = src.transform
        tile_shape = (src.height, src.width)
        
    aoi_mask = rasterize([path_2_tile_aoi(tile_path)], out_shape = tile_shape, fill=False, default_value=True, transform = transform)
    return aoi_mask