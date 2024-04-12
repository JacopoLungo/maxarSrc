import rasterio
from pathlib import Path
import numpy as np
from maxarseg.ESAM_segment import segment_utils
import pandas as pd
from maxarseg.polygonize import polygonize_with_values
import geopandas as gpd

def single_mask2Tif(tile_path, mask, out_name, out_dir_root = './output/tiff'):
    """
    Converts a binary mask to a GeoTIFF file.

    Args:
        tile_path (str): The path to the input tile file.
        mask (numpy.ndarray): The binary mask array.
        out_name (str): The name of the output GeoTIFF file.
        out_dir_root (str, optional): The root directory for the output GeoTIFF file. Defaults to './output/tiff'.

    Returns:
        None
    """
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
    
    out_meta.update({"driver": "GTiff",
                    "dtype": "uint8",
                    "count": 1})
    out_path = Path(out_dir_root) / out_name
    
    with rasterio.open(out_path, 'w', **out_meta) as dest:
            dest.write(mask, 1) 
    
    print(f"Mask written in {out_path}")
    
def masks2Tifs(tile_path , masks: np.ndarray, out_names: list, separate_masks: bool, out_dir_root = './output/tiff'):
    if not separate_masks: #merge the masks
        mask = segment_utils.merge_masks(masks)
        masks = np.expand_dims(mask, axis=0)
    
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
    
    out_meta.update({"driver": "GTiff",
                    "dtype": "uint8",
                    "count": 1})
    
    for j, out_name in enumerate(out_names):
        out_path = Path(out_dir_root) / out_name
        with rasterio.open(out_path, 'w', **out_meta) as dest:
                dest.write(masks[j], 1)   
        print(f"Mask written in {out_path}")
    
    return masks
    
def gen_names(tile_path, separate_masks=False):
    """
    Generate output file names based on the given tile path.

    Args:
        tile_path (Path): The path to the tile file.
        divide_masks (bool, optional): Whether to divide masks into separate files. Defaults to False.

    Returns:
        list: A list of output file names.
    """
    ev_name, tl_when, mos_name, tl_name = tile_path.parts[-4:]
    masks_names = ['road', 'tree', 'building']
    
    if separate_masks:
        out_names = [Path(ev_name) / tl_when / mos_name / (tl_name.split('.')[0] + '_' + mask_name + '.tif') for mask_name in masks_names]        
    else:
        out_names = [Path(ev_name) / tl_when / mos_name / (tl_name.split('.')[0] + '.tif')]
    
    return out_names

def masks2parquet(tile_path , masks: np.ndarray, out_names: list, out_dir_root = './output/tiff'):
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
        # convert no_overlap_masks to int
    tolerances = [0.001, 0.001, 0.001]
    no_overlap_masks_int = masks.astype(np.uint8)
    # polygonization
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
    gdf_list = []
    # cicling over the masks channels
    for i in range(no_overlap_masks_int.shape[0]):
        if no_overlap_masks_int[i].sum() != 0:
            gdf = polygonize_with_values(no_overlap_masks_int[i], class_id=i, tolerance=tolerances[i], transform=out_meta['transform'], crs=out_meta['crs'], pixel_threshold=10)
            gdf_list.append(gdf)
    # create a single gdf
    gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    # create gdf_first with the first row of gdf
    assert out_names.__len__() == 1, "Only one output name is allowed for parquet file"
    out_path = Path(out_dir_root) / out_names[0]
    # replace '.tif' with '.parquet'
    out_path = out_path.with_suffix('.parquet')
    gdf.to_parquet(out_path)
    return gdf