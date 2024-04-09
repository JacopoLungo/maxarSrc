import rasterio
from pathlib import Path
import numpy as np
from maxarseg.ESAM_segment import segment_utils

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