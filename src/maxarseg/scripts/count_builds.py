#%%
from maxarseg.assemble import holders
import os
os.chdir('/nfs/home/vaschetti/maxarSrc')
from pathlib import Path
from maxarseg.assemble import delimiters, names
import geopandas as gpd
import numpy as np
from maxarseg.samplers import samplers_utils
import time
import pandas as pd
# %%
def list_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
def list_tif_files(path):
    return [f for f in os.listdir(path) if f.endswith('.tif') and os.path.isfile(os.path.join(path, f))]

def list_parquet_files(path):
    return [f for f in os.listdir(path) if f.endswith('.parquet') and os.path.isfile(os.path.join(path, f))]

def filter_gdf_vs_aois_gdf(proj_gdf, aois_gdf):
    num_hits = np.array([0]*len(proj_gdf))
    for geom in aois_gdf.geometry:
        hits = proj_gdf.intersects(geom)
        num_hits = num_hits + hits.values
    return proj_gdf[num_hits >= 1]

# %%
class Event_light:
    def __init__(self,
                name,
                maxar_root = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data',
                maxar_metadata_path = './metadata/from_github_maxar_metadata/datasets',
                region = 'infer'):

        #Paths
        self.maxar_root = Path(maxar_root)
        self.buildings_ds_links_path = Path('./metadata/buildings_dataset_links.csv')
        self.maxar_metadata_path = Path(maxar_metadata_path)
        
        #Event
        self.name = name
        self.when = 'pre'
        self.region_name = names.get_region_name(self.name) if region == 'infer' else region
        self.bbox = delimiters.get_event_bbox(self.name, extra_mt=1000) #TODO può essere ottimizzata sfruttando i mosaici
        self.all_mosaics_names = names.get_mosaics_names(self.name, self.maxar_root, self.when)
        
        self.wlb_gdf = gpd.read_file('./metadata/eventi_confini.gpkg')
        self.filtered_wlb_gdf = self.wlb_gdf[self.wlb_gdf['event names'] == self.name]
        if self.filtered_wlb_gdf.iloc[0].geometry is None:
            print('Evento interamente su terra')
            self.cross_wlb = False
            self.filtered_wlb_gdf = None
        else:
            print('Evento su bordo')
            self.cross_wlb = True

        print(f'Creating event: {self.name}\nRegion: {self.region_name}\nMosaics: {self.all_mosaics_names}')
        #Roads
        self.road_gdf = None

        #Mosaics
        self.mosaics = {}

        #Init mosaics
        for m_name in self.all_mosaics_names:
            self.mosaics[m_name] = holders.Mosaic(m_name, self)
        
        self.total_tiles = sum([mosaic.tiles_num for mosaic in self.mosaics.values()])
        
    def __str__(self) -> str:
        res = f'\n_______________________________________________________\nEvent: {self.name}\nMosaics: {self.all_mosaics_names}\nTotal tiles: {self.total_tiles}\n_______________________________________________________\n'
        return res

#%%
root_folder = '/nfs/home/vaschetti/maxarSrc/output/03_05_19_45'
ev_names = list_directories(root_folder)
cols = {'event': [],
        'mosaic': [],
        'tile': [],
        'num_ms_build_aoi': [],
        'num_ms_build_aoi_no_water_for': [],
        'num_ms_build_aoi_no_water_sjoin': [],
        'parquet_build': [],
        'parquet_trees': []
        }
for ev_name in ev_names:
    mos_names = list_directories(os.path.join(root_folder, ev_name, 'pre'))
    if len(mos_names) > 0:
        event = Event_light(ev_name)
        print(event)
    for mos_name in mos_names:
        print(ev_name + '/' + mos_name)
        mos = event.mosaics[mos_name]
        print(mos)
        mos.set_build_gdf()
        print(f'len build gdf {len(mos.build_gdf):,}')
        tif_names = list_tif_files(os.path.join(root_folder, ev_name, 'pre', mos_name))
        parquets_names = list_parquet_files(os.path.join(root_folder, ev_name, 'pre', mos_name))
        print('proc_tifs', len(tif_names))
        print('proc parquet', len(parquets_names))
        i = 0
        for tile_name in tif_names:
            cols['event'].append(ev_name)
            cols['mosaic'].append(mos_name)
            cols['tile'].append(tile_name)
            
            tile_path = os.path.join(root_folder, ev_name, 'pre', mos_name, tile_name)
            parquet_path = tile_path[:-4] + '.parquet'
            #tile_aoi = gpd.GeoDataFrame({'geometry': [samplers_utils.path_2_tile_aoi(tile_path)]})
            
            time0 = time.time()
            num_aoi_build = len(mos.proj_build_gdf.iloc[mos.sindex_proj_build_gdf.query(samplers_utils.path_2_tile_aoi(tile_path))])
            cols['num_ms_build_aoi'].append(num_aoi_build)
            print('tile_aoi builds', num_aoi_build)
            print('aoi', time.time() - time0)
            print()
            
            time0 = time.time()
            tile_aoi_no_water = samplers_utils.path_2_tile_aoi_no_water(tile_path, event.filtered_wlb_gdf)
            num_ms_build_aoi_no_water_for = len(filter_gdf_vs_aois_gdf(mos.proj_build_gdf, tile_aoi_no_water))
            cols['num_ms_build_aoi_no_water_for'].append(num_ms_build_aoi_no_water_for)
            print('tile_aoi builds no water filter with for', num_ms_build_aoi_no_water_for)
            print(time.time() - time0)
            print()
            
            time0 = time.time()
            num_ms_build_aoi_no_water_sjoin = len(gpd.sjoin(mos.proj_build_gdf, tile_aoi_no_water, how='inner', op='intersects'))
            cols['num_ms_build_aoi_no_water_sjoin'].append(num_ms_build_aoi_no_water_sjoin)
            print('tile_aoi builds no water filter with sjoin', num_ms_build_aoi_no_water_sjoin)
            print(time.time() - time0)
            print()
            
            time0 = time.time()
            parquet_build = sum(pd.read_parquet(parquet_path, engine='pyarrow').class_id == 2)
            cols['parquet_build'].append(parquet_build)
            print('parquet build', parquet_build)
            print(time.time() - time0)
            print()
            print()
            
            

res_df = pd.DataFrame(cols)
