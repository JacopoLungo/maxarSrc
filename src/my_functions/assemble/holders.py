from pathlib import Path
from tqdm import tqdm
from time import time
import numpy as np
import rasterio
import warnings
import torch
import os

from rasterio.features import rasterize
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples

from my_functions.assemble import delimiters, filter, gen_gdf
from my_functions.ESAM_segment import segment, segment_utils
from my_functions.samplers import samplers, samplers_utils
from my_functions.geo_datasets import geoDatasets
from my_functions.configs import SegmentConfig
from my_functions.assemble import names
from my_functions.detect import detect
from my_functions import output

from my_functions import plotting_utils
import matplotlib.pyplot as plt

# Ignore all warnings
warnings.filterwarnings('ignore')

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
        print(f'Roads in {self.name} mosaic: {self.road_num}')
    
    def set_build_gdf(self):
        qk_hits = gen_gdf.intersecting_qks(*self.bbox)
        self.build_gdf = gen_gdf.qk_building_gdf(qk_hits, csv_path = self.event.buildings_ds_links_path)
        self.proj_build_gdf =  self.build_gdf.to_crs(self.crs)
        self.build_num = len(self.build_gdf)
    
    def __str__(self) -> str:
        return self.name
    
    def seg_road_tile(self, tile_path) -> np.array:
        seg_config = self.event.seg_config
        with rasterio.open(tile_path) as src:
            transform = src.transform
            tile_h = src.height
            tile_w = src.width
            #out_meta = src.meta.copy()
        query_bbox_poly = samplers_utils.path_2_tilePolygon(tile_path)
        road_lines = self.proj_road_gdf[self.proj_road_gdf.geometry.intersects(query_bbox_poly)]

        if len(road_lines) != 0:
            buffered_lines = road_lines.geometry.buffer(seg_config.road_width_mt)
            road_mask = rasterize(buffered_lines, out_shape=(tile_h, tile_w), transform=transform)
        else:
            print('No roads')
            road_mask = np.zeros((tile_h, tile_w))
        return road_mask  #shape: (h, w)
            
    def seg_tree_and_build_rnd_samples(self, tile_path):
        if self.build_gdf is None:
            self.set_build_gdf()
        
        seg_config = self.event.seg_config
        
        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path))
        sampler = samplers.MyRandomGeoSampler(dataset, length = seg_config.batch_size, size=seg_config.size)
        dataloader = DataLoader(dataset , sampler=sampler, collate_fn=stack_samples)
        
        for batch_ix, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
            original_img_tsr = batch['image']
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8') #TODO: l'immagine viene convertita in numpy ma magari è meglio lasciarla in tensor

            #TREES
            #get the tree boxes in batches and the number of trees for each image
            tree_boxes_b, num_trees4img = detect.get_GD_boxes(img_b,
                                                seg_config.GD_model,
                                                seg_config.TEXT_PROMPT,
                                                seg_config.BOX_THRESHOLD,
                                                seg_config.TEXT_THRESHOLD,
                                                dataset.res,
                                                device = seg_config.device,
                                                max_area_mt2 = seg_config.max_area_GD_boxes_mt2)
            
            #BUILDINGS
            #get the building boxes in batches and the number of buildings for each image
            building_boxes_b, num_build4img = detect.get_batch_buildings_boxes(batch['bbox'],
                                                                        proj_buildings_gdf = self.proj_build_gdf,
                                                                        dataset_res = dataset.res,
                                                                        ext_mt = 10)
                        
            max_detect = max(num_trees4img + num_build4img)
            
            #print("\n__________________________")
            #print("Batch number: ", i)
            #print(f'Num detections in batch per img: {num_trees4img + num_build4img}')
            
            #obtain the right input for the ESAM model (trees + buildings)
            input_points, input_labels = segment_utils.get_input_pts_and_lbs(tree_boxes_b, building_boxes_b, max_detect)
            
            all_masks_b = segment.ESAM_from_inputs(original_img_tsr,
                                                    torch.from_numpy(input_points),
                                                    torch.from_numpy(input_labels),
                                                    efficient_sam = seg_config.efficient_sam,
                                                    device = seg_config.device,
                                                    num_parall_queries = seg_config.ESAM_num_parall_queries)
            
            #for each image, discern the masks in trees, buildings and padding
            patch_masks_b = segment_utils.discern_mode_smooth(all_masks_b, num_trees4img, num_build4img, mode = 'bchw') #(b, channel, h_patch, w_patch)
            
            patch_masks_b = np.greater_equal(patch_masks_b, 0) #turn logits into bool
            
            #plotting
            for img, masks, tree_boxes, building_boxes in zip(img_b, patch_masks_b, tree_boxes_b, building_boxes_b):
                fig, axs = plt.subplots(1, 2, figsize = (15, 15))
                #plot trees and build separately
                plotting_utils.show_img(img, ax=axs[0])
                plotting_utils.show_mask(masks[0], axs[0], rgb_color = (255, 18, 18), alpha = 0.4)
                plotting_utils.show_box(tree_boxes, axs[0], color='r', lw = 0.4)
                
                plotting_utils.show_img(img, ax = axs[1])
                plotting_utils.show_mask(masks[1], axs[1], rgb_color = (131, 220, 242), alpha = 0.4)
                plotting_utils.show_box(building_boxes, axs[1], color='b', lw = 0.4)
    
    def new_seg_tree_and_build_tile(self, tile_path, debug = True):
        """
        This method should segment trees and buildings of a tile
        It should have access to a gdf of trees and a gdf of buildings stored as polygon??
        It should write the segmented mask in a canvas and return it.

        debug = True, if you want to process and plot some #batch_size random samples and plot them
        """
        
        if self.build_gdf is None:
            self.set_build_gdf()
        
        #Here call function to compute all the tree boxes and store them in a gdf
        
        
        
        seg_config = self.event.seg_config

        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path))
        if debug:
            sampler = samplers.MyRandomGeoSampler(dataset, length = seg_config.batch_size, size=seg_config.size)
            dataloader = DataLoader(dataset , sampler=sampler, collate_fn=stack_samples)

        else:
            sampler = samplers.BatchGridGeoSampler(dataset, batch_size=seg_config.batch_size, size=seg_config.size, stride=seg_config.stride)
            dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)

        canvas = np.full((3,) + samplers_utils.tile_sizes(dataset), fill_value = float('-inf') ,dtype=np.float32) #dim (3, h_tile, w_tile). The dim 0 is: tree, build, pad

        #all_batches_img_ixs = np.arange(len(sampler)*seg_config.batch_size).reshape((-1, seg_config.batch_size))
        #_, total_cols = sampler.get_num_rows_cols()
                
        #init TIMERS
        GD_total = build_box_total = Esam_total = post_proc_total = 0
        
        start_time_all = time()
        for batch_ix, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
            original_img_tsr = batch['image']
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8') #TODO: l'immagine viene convertita in numpy ma magari è meglio lasciarla in tensor

            #trees
            #GD_t_0 = time()
            
            #get the tree boxes in batches and the number of trees for each image
            GD_start = time()
            tree_boxes_b, num_trees4img = detect.get_GD_boxes(img_b,
                                                seg_config.GD_model,
                                                seg_config.TEXT_PROMPT,
                                                seg_config.BOX_THRESHOLD,
                                                seg_config.TEXT_THRESHOLD,
                                                dataset.res,
                                                device = seg_config.device,
                                                max_area_mt2 = seg_config.max_area_GD_boxes_mt2)
            GD_total += time() - GD_start
            #tree_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di tree boxes
            
            #print('GD_time: ', time() - GD_t_0)

            #get the building boxes in batches and the number of buildings for each image
            build_box_start = time()
            building_boxes_b, num_build4img = detect.get_batch_buildings_boxes(batch['bbox'],
                                                                        proj_buildings_gdf = self.proj_build_gdf,
                                                                        dataset_res = dataset.res,
                                                                        ext_mt = 10)
            build_box_total += time() - build_box_start
            
            #building_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di building boxes
            
            max_detect = max(num_trees4img + num_build4img)
            
            #print("\n__________________________")
            #print("Batch number: ", i)
            #print(f'Num detections in batch per img: {num_trees4img + num_build4img}')
            
            #obtain the right input for the ESAM model (trees + buildings)
            input_points, input_labels = segment_utils.get_input_pts_and_lbs(tree_boxes_b, building_boxes_b, max_detect)
            
            # segment the image and get for each image as many masks as the number of boxes,
            # for GPU constraint use num_parall_queries
            ESAM_start = time()
            all_masks_b = segment.ESAM_from_inputs(original_img_tsr,
                                                    torch.from_numpy(input_points),
                                                    torch.from_numpy(input_labels),
                                                    efficient_sam = seg_config.efficient_sam,
                                                    device = seg_config.device,
                                                    num_parall_queries = seg_config.ESAM_num_parall_queries)
            Esam_total += time() - ESAM_start
            
            #for each image, discern the masks in trees, buildings and padding
            post_proc_start = time()
            patch_masks_b = segment_utils.discern_mode_smooth(all_masks_b, num_trees4img, num_build4img, mode = 'bchw') #(b, channel, h_patch, w_patch)
            
            #se smooth = False le logits vengono trasformate in bool in discern_mode e quindi write_canvas si aspetta le bool
            #se smooth = True le logits vengono scritti direttamente in canvas e devi trasformarle in bool dopo
            
            if debug:
                patch_masks_b = np.greater_equal(patch_masks_b, 0) #turn logits into bool
                for img, masks, tree_boxes, building_boxes in zip(img_b, patch_masks_b, tree_boxes_b, building_boxes_b):
                    fig, axs = plt.subplots(1, 2, figsize = (15, 15))
                    #plot trees and build separately
                    plotting_utils.show_img(img, ax=axs[0])
                    plotting_utils.show_mask(masks[0], axs[0], rgb_color = (255, 18, 18), alpha = 0.4)
                    plotting_utils.show_box(tree_boxes, axs[0], color='r', lw = 0.4)
                    
                    plotting_utils.show_img(img, ax = axs[1])
                    plotting_utils.show_mask(masks[1], axs[1], rgb_color = (131, 220, 242), alpha = 0.4)
                    plotting_utils.show_box(building_boxes, axs[1], color='b', lw = 0.4)
                
            else:
                canvas = segment_utils.write_canvas_geo(canvas = canvas,
                                                        patch_masks_b =  patch_masks_b,
                                                        top_lft_indexes = batch['top_lft_index'],
                                                        smooth=seg_config.smooth_patch_overlap)

            post_proc_total += time() - post_proc_start
                
            if batch_ix%100 == 0 and batch_ix != 0:
                print('Avg times (sec/batch)')
                print(f'- GD: {(GD_total/(batch_ix + 1)):.4f}')
                print(f'- build_box: {(build_box_total/(batch_ix + 1)):.4f}')
                print(f'- ESAM: {(Esam_total/(batch_ix + 1)):.4f}')
                print(f'- post_proc: {(post_proc_total/(batch_ix + 1)):.4f}')
                
                #TODO: aggiundere qui un metodo per debug che ti fa vedere una patch con segmentazione e boxes
            #if batch_ix == 50:
            #    break
            
        if not debug:
            canvas = np.greater_equal(canvas, 0) #turn logits into bool
            print(f'\nTotal Time for {seg_config.batch_size * (batch_ix + 1)} images: ', time() - start_time_all)
            return canvas
    
    def seg_tree_and_build_tile(self, tile_path):
        if self.build_gdf is None:
            self.set_build_gdf()
        
        seg_config = self.event.seg_config

        dataset = geoDatasets.MxrSingleTile(str(tile_path))
        sampler = samplers.WholeTifGridGeoSampler(dataset, batch_size=seg_config.batch_size, size=seg_config.size, stride=seg_config.stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)

        canvas = np.zeros((3,) + samplers_utils.tile_sizes(dataset), dtype=np.uint8) #dim (3, h_tile, w_tile). The dim 0 is: tree, build, pad
        
        all_batches_img_ixs = np.arange(len(sampler)*seg_config.batch_size).reshape((-1, seg_config.batch_size))
        _, total_cols = sampler.get_num_rows_cols()
                
        #TIMERS
        GD_total = 0
        build_box_total = 0
        Esam_total = 0
        post_proc_total = 0
        
        start_time_all = time()
        for batch_ix, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
            original_img_tsr = batch['image']
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8') #TODO: l'immagine viene convertita in numpy ma magari è meglio lasciarla in tensor

            #trees
            #GD_t_0 = time()
            
            #get the tree boxes in batches and the number of trees for each image
            GD_start = time()
            tree_boxes_b, num_trees4img = detect.get_GD_boxes(img_b,
                                                seg_config.GD_model,
                                                seg_config.TEXT_PROMPT,
                                                seg_config.BOX_THRESHOLD,
                                                seg_config.TEXT_THRESHOLD,
                                                dataset.res,
                                                device = seg_config.device,
                                                max_area_mt2 = seg_config.max_area_GD_boxes_mt2)
            GD_total += time() - GD_start
            #tree_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di tree boxes
            
            #print('GD_time: ', time() - GD_t_0)

            #get the building boxes in batches and the number of buildings for each image
            build_box_start = time()
            building_boxes_b, num_build4img = detect.get_batch_buildings_boxes(batch['bbox'],
                                                                        proj_buildings_gdf = self.proj_build_gdf,
                                                                        dataset_res = dataset.res,
                                                                        ext_mt = 10)
            build_box_total += time() - build_box_start
            
            #building_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di building boxes
            
            max_detect = max(num_trees4img + num_build4img)
            
            #print("\n__________________________")
            #print("Batch number: ", i)
            #print(f'Num detections in batch per img: {num_trees4img + num_build4img}')
            
            #obtain the right input for the ESAM model (trees + buildings)
            input_points, input_labels = segment_utils.get_input_pts_and_lbs(tree_boxes_b, building_boxes_b, max_detect)
            
            # segment the image and get for each image as many masks as the number of boxes,
            # for GPU constraint use num_parall_queries
            ESAM_start = time()
            all_masks_b = segment.ESAM_from_inputs(original_img_tsr,
                                                    torch.from_numpy(input_points),
                                                    torch.from_numpy(input_labels),
                                                    efficient_sam = seg_config.efficient_sam,
                                                    device = seg_config.device,
                                                    num_parall_queries = seg_config.ESAM_num_parall_queries)
            Esam_total += time() - ESAM_start
            
            
            #for each image, discern the masks in trees, buildings and padding
            post_proc_start = time()
            patch_masks_b = segment_utils.discern_mode(all_masks_b, num_trees4img, num_build4img, mode = 'bchw')
            
            canvas = segment_utils.write_canvas(canvas = canvas,
                                                patch_masks_b =  patch_masks_b,
                                                img_ixs = all_batches_img_ixs[batch_ix],
                                                stride = seg_config.stride,
                                                total_cols = total_cols)
            
            post_proc_total += time() - post_proc_start
            
            if batch_ix%100 == 0:
                print('Avg times (sec/batch)')
                print(f'- GD: {(GD_total/(batch_ix + 1)):.4f}')
                print(f'- build_box: {(build_box_total/(batch_ix + 1)):.4f}')
                print(f'- ESAM: {(Esam_total/(batch_ix + 1)):.4f}')
                print(f'- post_proc: {(post_proc_total/(batch_ix + 1)):.4f}')
                #TODO: aggiundere qui un metodo per debug che ti fa vedere una patch con segmentazione e boxes
            
        print(f'\nTotal Time for {seg_config.batch_size * (batch_ix + 1)} images: ', time() - start_time_all)
        return canvas

    
    def segment_tile(self, tile_path, out_dir_root, overwrite = False):
                        
        if self.build_gdf is None:
            self.set_build_gdf()
        if self.road_gdf is None:
            self.set_road_gdf()
        
        tile_path = Path(tile_path)
        out_dir_root = Path(out_dir_root)
        
        ev_name, tl_when, mos_name, tl_name = tile_path.parts[-4:]
        masks_names = ['road', 'tree', 'building']
        out_names = [Path(ev_name) / tl_when / mos_name / (tl_name.split('.')[0] + '_' + mask_name + '.tif') for mask_name in masks_names]        
        
        (out_dir_root / out_names[0]).parent.mkdir(parents=True, exist_ok=True) #create folder if not exists

        if not overwrite:
            for out_name in out_names:
                assert not (out_dir_root / out_name).exists(), f'File {out_name} already exists'
        
        tree_and_build_mask = self.seg_tree_and_build_tile(tile_path)
        road_mask = self.seg_road_tile(tile_path)
        
        #TODO: aggiungere post processing mask (tappare buchi, cancellare paritcelle)
        overlap_masks = np.concatenate((np.expand_dims(road_mask, axis=0), tree_and_build_mask[:-1]) , axis = 0)
        
        no_overlap_masks = segment_utils.rmv_mask_overlap(overlap_masks)
        
        for j, out_name in enumerate(out_names):
            output.single_mask2Tif(tile_path, no_overlap_masks[j], out_name = out_name, out_dir_root = out_dir_root)
    
    def segment_all_tiles(self):
        for tile_path in self.tiles_paths:
            self.segment_tile(tile_path) 


class Event:
    def __init__(self,
                 name,
                 seg_config: SegmentConfig,
                 when = 'pre', #'pre', 'post', None or 'None'
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
    
    #Segment methods
    def seg_all_mosaics(self):
        for __, mosaic in self.mosaics.items():
            mosaic.segment_all_tiles()