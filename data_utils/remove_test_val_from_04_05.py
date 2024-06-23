
import os
import pandas as pd
import numpy as np
import shutil
import glob

def count_tif_parquet(folder_path):
    count_tif = 0
    count_parquet = 0
    for root, dirs, files in os.walk(folder_path):
        count_tif += len(glob.glob(os.path.join(root, '*.tif')))
        count_parquet += len(glob.glob(os.path.join(root, '*.parquet')))
    return count_tif, count_parquet

root = '/nfs/projects/overwatch/maxar-segmentation/outputs/04_05'
bin_path = '/nfs/projects/overwatch/maxar-segmentation/outputs/from_04_05_duplicated_test_val'
i = 0
def find_tif_images(folder_path):
    print('Searching for tif files in:', folder_path)
    tif_files = []
    for root, dirs, files in os.walk(folder_path):
        tif_files.extend(glob.glob(os.path.join(root, '*.tif')))
    eventi_da_non_spostare = ['shovi-georgia-landslide-8Aug23', 'India-Floods-Oct-2023', 'Kalehe-DRC-Flooding-5-8-23', 'NWT-Canada-Aug-23']
    tif_files = [tif_file.replace(folder_path, '') for tif_file in tif_files]
    tif_files = [tif_file[1:] for tif_file in tif_files if tif_file.split('/')[1] not in eventi_da_non_spostare]
    return tif_files

for partition in ['test', 'val']:
    img_2_move = find_tif_images(os.path.join(root, partition))
    for img in img_2_move:
        source_path = os.path.join(root, 'train', img)
        if os.path.exists(source_path):
            shutil.move(source_path, os.path.join(bin_path, partition))
            print('Moved:', source_path)
        else:
            print('File not found:', source_path)
            
        parquet_source_path = source_path.replace('.tif', '.parquet')
        if os.path.exists(parquet_source_path):
            shutil.move(parquet_source_path, os.path.join(bin_path, partition))
            print('Moved:', parquet_source_path)
        else:
            print('File not found:', parquet_source_path)