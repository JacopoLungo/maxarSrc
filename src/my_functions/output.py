import rasterio

def single_mask2Tif(tile_path, mask, out_name, out_path_root = '/home/vaschetti/maxarSrc/output'):
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
    
    out_meta.update({"driver": "GTiff",
                    "dtype": "uint8",
                    "count": 1})
    out_path = out_path_root + '/'+ out_name
    with rasterio.open(out_path, 'w', **out_meta) as dest:
            dest.write(mask, 1) 
    
    print(f"Mask written in {out_path}")