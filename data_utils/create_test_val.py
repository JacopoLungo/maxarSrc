import os
os.chdir('/nfs/home/vaschetti/projects/igarss_2024/maxarSrc/')
from maxarseg.assemble import names
import pandas as pd
import numpy as np
import shutil

ev_names = names.get_all_events()
root = '/nfs/home/vaschetti/projects/igarss_2024/maxarSrc/stats/'
i = 0
dfs = []
npys = []
for name in sorted(ev_names):
    df_path = os.path.join(root, name + '_lbl_stats.csv')
    npy_path = os.path.join(root, name + '_high_res_entropies.npy')
    if os.path.exists(df_path):
        dfs.append(pd.read_csv(df_path))
        npys.append(np.load(npy_path))
        print(name)

global_df = pd.concat(dfs, ignore_index=True)
global_entropies = np.concatenate(npys, axis = 0)

ev_names = names.get_all_events()
root_train = '/nfs/projects/overwatch/maxar-segmentation/outputs/04_05/train'
root_test = '/nfs/projects/overwatch/maxar-segmentation/outputs/04_05/test/'
root_val = '/nfs/projects/overwatch/maxar-segmentation/outputs/04_05/val/'
for ev_name in sorted(ev_names):
    try:
        partial_df = global_df[global_df['event'] == ev_name].sort_values(by='entropy', ascending=False)
        text_ix = 3
        partial_path_selected_tile_to_test = os.path.join(partial_df.iloc[text_ix]['event'],'pre', partial_df.iloc[text_ix]['mosaic'], partial_df.iloc[text_ix]['tile']) #la 3 img per il test
        val_ix = 4
        partial_path_selected_tile_to_val = os.path.join(partial_df.iloc[val_ix]['event'],'pre', partial_df.iloc[val_ix]['mosaic'], partial_df.iloc[val_ix]['tile']) #la 4 img per il val
        
        #TEST
        print('Now moving to test')
        #TIF
        source_path_test = os.path.join(root_train, partial_path_selected_tile_to_test)
        print(os.path.exists(source_path_test), source_path_test)
        destination_path_test = os.path.join(root_test, partial_path_selected_tile_to_test)
        print(os.path.exists(destination_path_test), destination_path_test)
        
        #Parquet
        source_path_test_parquet = source_path_test[:-4] + '.parquet'
        print(os.path.exists(source_path_test_parquet), source_path_test_parquet)
        destination_path_test_parquet = destination_path_test[:-4] + '.parquet'
        print(os.path.exists(destination_path_test_parquet), destination_path_test_parquet)
        
        #create destination folders
        os.makedirs(os.path.dirname(destination_path_test), exist_ok=True)        
        
        if os.path.exists(source_path_test):
            shutil.move(source_path_test, os.path.dirname(destination_path_test))
            print('tif moved')
        
        if os.path.exists(source_path_test_parquet):
            shutil.move(source_path_test_parquet, os.path.dirname(destination_path_test_parquet))
            print('parquet moved')
            
        print(os.path.exists(source_path_test), source_path_test)
        print(os.path.exists(destination_path_test), destination_path_test)
        print(os.path.exists(source_path_test_parquet), source_path_test_parquet)
        print(os.path.exists(destination_path_test_parquet), destination_path_test_parquet)
        
        #VAL
        print('Now moving to val')
        #TIF
        source_path_val = os.path.join(root_train, partial_path_selected_tile_to_val)
        print(os.path.exists(source_path_val), source_path_val)
        destination_path_val = os.path.join(root_val, partial_path_selected_tile_to_val)
        print(os.path.exists(destination_path_val), destination_path_val)
        
        #Parquet
        source_path_val_parquet = source_path_val[:-4] + '.parquet'
        print(os.path.exists(source_path_val_parquet), source_path_val_parquet)
        destination_path_val_parquet = destination_path_val[:-4] + '.parquet'
        print(os.path.exists(destination_path_val_parquet), destination_path_val_parquet)
        
        #create destination folders
        os.makedirs(os.path.dirname(destination_path_val), exist_ok=True)
        
        if os.path.exists(source_path_val):
            shutil.move(source_path_val, os.path.dirname(destination_path_val))
            
        if os.path.exists(source_path_val_parquet):
            shutil.move(source_path_val_parquet, os.path.dirname(destination_path_val_parquet))
            
        print(os.path.exists(source_path_val), source_path_val)
        print(os.path.exists(destination_path_val), destination_path_val)
        print(os.path.exists(source_path_val_parquet), source_path_val_parquet)
        print(os.path.exists(destination_path_val_parquet), destination_path_val_parquet)
        
    except Exception as e:
        print(ev_name)
        print(e)