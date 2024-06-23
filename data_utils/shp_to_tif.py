#%%
import rasterio
import os
import geopandas as gpd
from rasterio.features import rasterize
import numpy as np
import sys
import shutil
import pandas as pd


def find_shapefiles(directory):
    shapefiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_checked.shp"):
                full_path = os.path.join(root, file)
                shapefiles.append(full_path)
    return shapefiles

def find_checked(dir):
    counter = 0
    
    for root, dirs, files in os.walk(dir):
        root_flag = False
        for file in files:
            if file.endswith("_checked.shp"):
                counter += 1
                root_flag = True
        if not root_flag and 'shapes' in root:
            print(f"No checked shapefile in {root}")
    return counter

def find_file(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if "southafrica-flooding22_9216_7680_512" in file:
                print(os.path.join(root, file))
#%%

def main():
    rasterized_shp = 0
    out_root = './labelled_45'
    
    parent_dir = './labelled_45/labelled_45'
    shapefile_paths = find_shapefiles(parent_dir)
    
    for shapefile_path in shapefile_paths:
        #path = '/nfs/home/vaschetti/projects/igarss_2024/maxarSrc/esempio_shape_gt_label/SouthAfrica_1/annotation_south_africa_1.shp'
        general_file_name = os.path.splitext(os.path.basename(shapefile_path))[0].replace('_checked', '')
        shapes_gdf = gpd.read_file(shapefile_path)
        shapes_gdf['category_id'] = pd.Categorical(shapes_gdf['class_id'], categories=[0, 2, 1], ordered=True)
        shapes_gdf = shapes_gdf.sort_values(by='category_id')
        
        
        image_path = '/'.join(shapefile_path.split('/')[:-2]) +'/'+ general_file_name+ '_img.tif'
        pseudo_lbl_path = image_path.replace('_img.tif', '_pseudo_lbl.tif')
        
        try:
            #TODO: this tif file shuold have the right path
            with rasterio.open(image_path) as src:
                width, height = src.width, src.height
                if (width != height) or height != 512:
                    raise ValueError('The image is not 512x512')
                transform = src.transform
                out_meta = src.meta.copy()

            shapes = ((geom, value) for geom, value in zip(shapes_gdf.geometry, shapes_gdf['class_id']))
            rasterized = rasterize(shapes, out_shape=(width, height), transform=transform, all_touched=True, fill = 255, dtype='uint8')
            rasterized = rasterized.astype(np.uint8)

            out_meta.update({"driver": "GTiff",
                            "dtype": "uint8",
                            "count": 1})

            out_path = os.path.join(out_root, 'gt_45', general_file_name+'_gt.tif')
            with rasterio.open(out_path, 'w', **out_meta) as dest:
                dest.write(rasterized, 1)
                print(f"Mask written in {out_path}")
            
            shutil.copy(image_path, os.path.join(out_root, 'img_45', general_file_name+'_img.tif'))
            shutil.copy(pseudo_lbl_path, os.path.join(out_root, 'pseudo_lbl_45', general_file_name+'_pseudo_lbl.tif'))
            rasterized_shp += 1
        except Exception as e:
            print(f"Error in {shapefile_path}: {e}")
            continue
    print(f"Rasterized {rasterized_shp} shapefiles")

if __name__ == '__main__':

    main()
    
    
#python tools/shp_to_tif.py ./labelled_10/labelled(yw)