import rasterio
from pathlib import Path

def single_mask2Tif(tile_path, mask, out_name, out_dir_root = '/home/vaschetti/maxarSrc/output/tiff'):
    """
    Input:
    tile_path: str, path to the tile
    mask: np.array, mask to be written, dimension (h, w)
    out_name: str, name of the output file
    out_dir_root: str, root of the output directory
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