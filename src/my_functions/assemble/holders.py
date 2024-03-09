from my_functions.assemble import delimiters
from my_functions.configs import SegmentConfig
from pathlib import Path
from my_functions import samplers
from tqdm import tqdm
import numpy as np
import torch

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

        dataset = geoDatasets.MxrSingleTile(str(tile_path))
        sampler = samplers.WholeTifGridGeoSampler(dataset, batch_size=seg_config.batch_size, size=seg_config.size, stride=seg_config.stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)

        canvas = np.zeros((3,) + samplers_utils.tile_sizes(dataset), dtype=np.uint8) #dim (3, h_tile, w_tile). The first dim is tree, build, pad
        
        all_batches_img_ixs = np.arange(len(sampler)).reshape((-1, batch_size))
        _, total_cols = sampler.get_num_rows_cols()
        
        i = 0
        f_i = 50
        start_time_all = time()
        for batch_ix, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
            i+=1
            original_img_tsr = batch['image']
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8') #TODO: l'immagine viene convertita in numpy ma magari è meglio lasciarla in tensor

            #trees
            GD_t_0 = time()
            
            #get the tree boxes in batches and the number of trees for each image
            tree_boxes_b, num_trees4img = detect.get_GD_boxes(img_b,
                                                seg_config.GD_model,
                                                seg_config.TEXT_PROMPT,
                                                seg_config.BOX_THRESHOLD,
                                                seg_config.TEXT_THRESHOLD,
                                                dataset.res,
                                                device = seg_config.device,
                                                max_area_mt2 = seg_config.max_area_GD_boxes_mt2)
            #tree_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di tree boxes
            
            #print('GD_time: ', time() - GD_t_0)

            #get the building boxes in batches and the number of buildings for each image
            building_boxes_b, num_build4img = detect.get_batch_buildings_boxes(batch['bbox'],
                                                                        proj_buildings_gdf = self.proj_build_gdf,
                                                                        dataset_res = dataset.res,
                                                                        ext_mt = 10)
            
            #building_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di building boxes
            
            max_detect = max(num_trees4img + num_build4img)
            
            #print("\n__________________________")
            #print("Batch number: ", i)
            #print(f'Num detections in batch per img: {num_trees4img + num_build4img}')
            
            #obtain the right input for the ESAM model (trees + buildings)
            input_points, input_labels = segment_utils.get_input_pts_and_lbs(tree_boxes_b, building_boxes_b, max_detect)
            
            # segment the image and get for each image as many masks as the number of boxes,
            # for GPU constraint use num_parall_queries
            all_masks_b = segment.ESAM_from_inputs(original_img_tsr,
                                                    torch.from_numpy(input_points),
                                                    torch.from_numpy(input_labels),
                                                    efficient_sam = seg_config.efficient_sam,
                                                    device = seg_config.device,
                                                    num_parall_queries = 5)
            
            
            #for each image, discern the masks in trees, buildings and padding
            patch_masks_b = segment_utils.discern_mode(all_masks_b, num_trees4img, num_build4img, mode = 'bchw')
            
            canvas = segment_utils.write_canvas(canvas = canvas,
                                                patch_masks_b =  patch_masks_b,
                                                img_ixs = all_batches_img_ixs[batch_ix],
                                                stride = seg_config.stride,
                                                total_cols = total_cols)
            
            if i == f_i:
                break
            
        #print(f'\nTotal Time for {seg_config.batch_size * i} images: ', time() - start_time_all)
        return canvas
        
    
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