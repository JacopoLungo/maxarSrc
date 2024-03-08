from my_functions.assemble import delimiters
from my_functions.configs import SegmentConfig
from pathlib import Path
from my_functions import samplers
from tqdm import tqdm
import numpy as np

#from my_functions.segment import segment
from my_functions.detect import detect

import rasterio
from rasterio.features import rasterize
from my_functions.samplers.samplers_utils import path_2_tilePolygon
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples

from my_functions.assemble import filter
from my_functions.assemble import gen_gdf
from my_functions.geo_datasets import geoDatasets
from my_functions.assemble import names


class Mosaic:
    def __init__(self,
                 name,
                 event
                 ):
        
        #Mosaic
        self.name = name
        self.event = event
        self.bbox, self.crs = delimiters.get_mosaic_bbox(self.event.name,
                                          self.name,
                                          self.event.maxar_metadata_path,
                                          extra_mt=1000)
        
        self.when = list((self.event.maxar_root / self.event.name).glob('**/*'+self.name))[0].parts[-2]
        self.tiles_paths = list((self.event.maxar_root / self.event.name / self.when / self.name).glob('*.tif'))
        self.tiles_num = len(self.tiles_paths)

        #Roads
        self.road_gdf = None
        self.proj_road_gdf = None
        self.road_num = None

        #Buildings
        self.build_gdf = None
        self.proj_build_gdf = None
        self.build_num = None

    def set_road_gdf(self):
        if self.event.road_gdf is None:
            self.event.set_road_gdf()

        self.road_gdf = filter.filter_gdf_w_bbox(self.event.road_gdf, self.bbox)
        self.proj_road_gdf =  self.road_gdf.to_crs(self.crs)
        self.road_num = len(self.road_gdf)
    
    def set_build_gdf(self):
        qk_hits = gen_gdf.intersecting_qks(*self.bbox)
        self.build_gdf = gen_gdf.qk_building_gdf(qk_hits, csv_path = self.event.buildings_ds_links_path)
        self.proj_build_gdf =  self.build_gdf.to_crs(self.crs)
        self.build_num = len(self.build_gdf)
    
    def __str__(self) -> str:
        return self.name
    
    def get_tile_road_mask_np(self, tile_path, ext_mt = 10):
        with rasterio.open(tile_path) as src:
            transform = src.transform
            tile_h = src.height
            tile_w = src.width
            out_meta = src.meta.copy()
        query_bbox_poly = path_2_tilePolygon(tile_path)
        road_lines = self.proj_road_gdf[self.proj_road_gdf.geometry.intersects(query_bbox_poly)]

        if len(road_lines) != 0:
            buffered_lines = road_lines.geometry.buffer(ext_mt)
            road_mask = rasterize(buffered_lines, out_shape=(tile_h, tile_w), transform=transform)
        else:
            print('No roads')
            road_mask = np.zeros((tile_h, tile_w))
        return road_mask
            
    def segment_tile(self, tile_path):
        seg_config = self.event.seg_config

        dataset = geoDatasets.Maxar(str(tile_path))
        sampler = samplers.MyBatchGridGeoSampler(dataset, batch_size=seg_config.batch_size, size=seg_config.size, stride=seg_config.stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)

        canvas = np.zeros((seg_config.size, seg_config.size, 3), dtype=np.uint8)

        for batch in tqdm(dataloader):          
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8')
            tree_boxes_b, num_trees4img = detect.get_GD_boxes(img_b,
                                                seg_config.GD_model,
                                                seg_config.TEXT_PROMPT,
                                                seg_config.BOX_TRESHOLD,
                                                seg_config.TEXT_TRESHOLD,
                                                dataset.res,
                                                max_area_mt2 = seg_config.max_area_GD_boxes_mt2)
            
            building_boxes_b, num_build4img = segment.get_batch_buildings_boxes(batch['bbox'],
                                                                proj_buildings_gdf = self.proj_build_gdf,
                                                                dataset_res = dataset.res,
                                                                ext_mt = 10)
            tree_and_building_mask_b = None
            road_mask_b = None
            all_mask_b = None


            #fig, axs = plt.subplots(1, batch_size, figsize=(30, 30))
            #for i in range(batch_size):
            #    axs[i].imshow(img_b[i])
            #print(img_b.shape)
        
        #TODO: salvare la canvas come tiff
        
    

    def segment_all_tiles(self):
        for tile_path in self.tiles_paths:
            self.segment_tile(tile_path)


class Event:
    def __init__(self,
                 name,
                 seg_config: SegmentConfig,
                 when = 'pre', #'pre', 'post' or None
                 maxar_root = '/mnt/data2/vaschetti_data/maxar',
                 maxar_metadata_path = '/home/vaschetti/maxarSrc/metadata/from_github_maxar_metadata/datasets',
                 region = 'infer'
                 ):
        #Segmentation
        self.seg_config = seg_config

        #Paths
        self.maxar_root = Path(maxar_root)
        self.buildings_ds_links_path = Path('/home/vaschetti/maxarSrc/metadata/buildings_dataset_links.csv')
        self.maxar_metadata_path = Path(maxar_metadata_path)
        
        #Event
        self.name = name
        self.when = when
        self.region_name = names.get_region_name(self.name) if region == 'infer' else region
        self.bbox = delimiters.get_event_bbox(self.name, extra_mt=1000) #TODO può essere ottimizzata sfruttando i mosaici
        self.all_mosaics_names = names.get_mosaics_names(self.name, self.maxar_root, self.when)
    
        #Roads
        self.road_gdf = None

        #Mosaics
        self.mosaics = {}

        #Init mosaics
        for m_name in self.all_mosaics_names:
            self.mosaics[m_name] = Mosaic(m_name, self)


    #Roads methods
    def set_road_gdf(self): #set road_gdf for the event
        region_road_gdf = gen_gdf.get_region_road_gdf(self.region_name)
        self.road_gdf = filter.filter_gdf_w_bbox(region_road_gdf, self.bbox)

    def set_mos_road_gdf(self, mosaic_name): #set road_gdf for the mosaic
        if self.road_gdf is None:
            self.set_road_gdf()

        self.mosaics[mosaic_name].set_road_gdf()

    def set_all_mos_road_gdf(self): #set road_gdf for all the mosaics
        for mosaic_name, mosaic in self.mosaics.items():
            if mosaic.road_gdf is None:
                self.set_mos_road_gdf(mosaic_name)
    
    #Buildings methods
    def set_build_gdf_in_mos(self, mosaic_name):
        self.mosaics[mosaic_name].set_build_gdf()

    def set_build_gdf_all_mos(self):
        for mosaic_name, mosaic in self.mosaics.items():
            if mosaic.build_gdf is None:
                self.set_build_gdf_in_mos(mosaic_name)

    def get_mosaic(self, mosaic_name):
        return self.mosaics[mosaic_name]