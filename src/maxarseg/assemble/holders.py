#Generic
from pathlib import Path
from tqdm import tqdm

from time import time, perf_counter
import numpy as np
import rasterio
from rasterio.features import rasterize
import warnings
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
import torchvision
import geopandas as gpd
import supervision as sv
from typing import Tuple
import matplotlib.pyplot as plt
import os
import sys

#My functions
from maxarseg.assemble import delimiters, filter, gen_gdf, names
from maxarseg.ESAM_segment import segment, segment_utils
from maxarseg.samplers import samplers, samplers_utils
from maxarseg.geo_datasets import geoDatasets
from maxarseg.configs import SegmentConfig, DetectConfig
from maxarseg.detect import detect, detect_utils
from maxarseg import output
from maxarseg import plotting_utils

#GroundingDino
from groundingdino.util.inference import load_model as GD_load_model
from groundingdino.util.inference import predict as GD_predict

#Deep forest
from deepforest import main

#Evit
from efficientvit.sam_model_zoo import create_sam_model
from segment_anything.utils.transforms import ResizeLongestSide
from maxarseg.EVIT_segment import evit_utils
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from maxarseg.EVIT_segment import evit_segment

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
                                                        extra_mt = 1000)
        
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

    def __str__(self) -> str:
        return self.name
    
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

        if len(self.build_gdf) == 0: #here use google buildings
            self.build_gdf = gen_gdf.google_building_gdf(event_name=self.event.name, bbox=self.bbox)
            if len(self.build_gdf) == 0:
                self.build_gdf = None
                self.proj_build_gdf = None
                print('No buildings found for this mosaic either in Ms Buildings or in Google Open Buildings')
                return False
                
        self.proj_build_gdf = self.build_gdf.to_crs(self.crs)
        
    def seg_road_tile(self, tile_path) -> np.ndarray:
        seg_config = self.event.seg_config
        with rasterio.open(tile_path) as src:
            transform = src.transform
            tile_h = src.height
            tile_w = src.width
            tile_shape = (tile_h, tile_w)
        
        tile_aoi = samplers_utils.path_2_tile_aoi(tile_path)
        aoi_mask = rasterize([tile_aoi], out_shape = tile_shape, fill=False, default_value=True, transform = transform)

            #out_meta = src.meta.copy()
        query_bbox_poly = samplers_utils.path_2_tile_aoi(tile_path)
        road_lines = self.proj_road_gdf[self.proj_road_gdf.geometry.intersects(query_bbox_poly)]

        if len(road_lines) != 0:
            buffered_lines = road_lines.geometry.buffer(seg_config.road_width_mt)
            road_mask = rasterize(buffered_lines, out_shape=(tile_h, tile_w), transform=transform)
            road_mask = np.where(aoi_mask, road_mask, False)
        else:
            print('No roads')
            road_mask = np.zeros((tile_h, tile_w))
        return road_mask  #shape: (h, w)
    
    def detect_trees_tile_DeepForest(self, tile_path) -> Tuple[np.ndarray, ...]:
        config = self.event.det_config
        model = main.deepforest(config_args = { 'devices' : config.DF_device,
                                                'retinanet': {'score_thresh': config.DF_box_threshold},
                                                'accelerator': 'cuda',
                                                'batch_size': config.DF_batch_size})
        model.use_release()
        
        boxes_df = model.predict_tile(tile_path,
                                    return_plot = False,
                                    patch_size = config.DF_patch_size,
                                    patch_overlap = config.DF_patch_overlap)
        
        
        boxes = boxes_df.iloc[:, :4].values
        score = boxes_df['score'].values
        
        del model
        torch.cuda.empty_cache()
        
        return boxes, score
    
    def detect_trees_tile_GD(self, tile_path) -> Tuple[np.ndarray, np.ndarray]:
        det_config = self.event.det_config
        
        #load model
        model = GD_load_model(det_config.CONFIG_PATH, det_config.WEIGHTS_PATH).to(det_config.device)
        print('\n- GD model device:', next(model.parameters()).device)
        
        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path))
        sampler = samplers.BatchGridGeoSampler(dataset, batch_size=det_config.GD_batch_size, size=det_config.size, stride=det_config.stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)
        
        glb_tile_tree_boxes = torch.empty(0, 4)
        all_logits = torch.empty(0)
        
        for batch in tqdm(dataloader, total = len(dataloader), desc="Detecting Trees with GDino"):
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8')
            
            for img, img_top_left_index in zip(img_b, batch['top_lft_index']):
                image_transformed = detect_utils.GD_img_load(img)
                tree_boxes, logits, phrases = GD_predict(model,
                                                         image_transformed,
                                                         det_config.TEXT_PROMPT,
                                                         det_config.BOX_THRESHOLD,
                                                         det_config.TEXT_THRESHOLD,
                                                         device = det_config.device)
                
                rel_xyxy_tree_boxes = detect_utils.GDboxes2SamBoxes(tree_boxes, img_shape = det_config.size)
                top_left_xy = np.array([img_top_left_index[1], #from an index to xyxy
                                        img_top_left_index[0],
                                        img_top_left_index[1],
                                        img_top_left_index[0]])
                
                #turn boxes from patch xyxy coords to global xyxy coords
                glb_xyxy_tree_boxes = rel_xyxy_tree_boxes + top_left_xy
                
                glb_tile_tree_boxes = np.concatenate((glb_tile_tree_boxes, glb_xyxy_tree_boxes))
                all_logits = np.concatenate((all_logits, logits))
        
        #del model and free GPU
        del model
        torch.cuda.empty_cache()
        
        return  glb_tile_tree_boxes, all_logits        
    
    def detect_trees_tile(self, tile_path, georef = True):
        with rasterio.open(tile_path) as src:
            to_xy = src.xy
            crs = src.crs
            
        config = self.event.det_config
        
        
        #TODO: uncomment if not debugging GD_glb_tile_tree_boxes, GD_scores = self.detect_trees_tile_GD(tile_path)
        deepForest_glb_tile_tree_boxes, deepForest_scores = self.detect_trees_tile_DeepForest(tile_path)
        
        #TODO: glb_tile_tree_boxes = np.concatenate((GD_glb_tile_tree_boxes, deepForest_glb_tile_tree_boxes))
        #TODO. glb_tile_tree_scores = np.concatenate((GD_scores, deepForest_scores))
        glb_tile_tree_boxes = deepForest_glb_tile_tree_boxes
        glb_tile_tree_scores = deepForest_scores
        
        print('Number of tree boxes before filtering: ', len(glb_tile_tree_boxes))
        
        det_config = self.event.det_config
        
        keep_ix_box_area = detect_utils.filter_on_box_area_mt2(glb_tile_tree_boxes,
                                                               max_area_mt2 = det_config.max_area_GD_boxes_mt2,
                                                               box_format = 'xyxy')
        glb_tile_tree_boxes = glb_tile_tree_boxes[keep_ix_box_area]
        glb_tile_tree_scores = glb_tile_tree_scores[keep_ix_box_area]
        print('boxes area filtering: ', len(keep_ix_box_area) - np.sum(keep_ix_box_area), 'boxes removed')
        
        keep_ix_box_ratio = detect_utils.filter_on_box_ratio(glb_tile_tree_boxes,
                                                             min_edges_ratio = det_config.min_ratio_GD_boxes_edges,
                                                             box_format = 'xyxy')
        glb_tile_tree_boxes = glb_tile_tree_boxes[keep_ix_box_ratio]
        glb_tile_tree_scores = glb_tile_tree_scores[keep_ix_box_ratio]
        print('box edge ratio filtering:', len(keep_ix_box_ratio) - np.sum(keep_ix_box_ratio), 'boxes removed')
        
        keep_ix_nms = torchvision.ops.nms(torch.tensor(glb_tile_tree_boxes), torch.tensor(glb_tile_tree_scores), config.nms_threshold)
        len_bf_nms = len(glb_tile_tree_boxes)
        glb_tile_tree_boxes = glb_tile_tree_boxes[keep_ix_nms]
        glb_tile_tree_scores = glb_tile_tree_scores[keep_ix_nms]
        print('nms filtering:', len_bf_nms - len(keep_ix_nms), 'boxes removed')
        
        print('Number of tree boxes after all filtering: ', len(glb_tile_tree_boxes))
        
        if georef: #create a gdf with the boxes in proj coordinates
            for i, box in enumerate(glb_tile_tree_boxes):
                #need to invert x and y to go from col row to row col index
                glb_tile_tree_boxes[i] = np.array(to_xy(box[1], box[0]) + to_xy(box[3], box[2]))
                
            cols = {'score': list(glb_tile_tree_scores),
                    'geometry': [samplers_utils.xyxyBox2Polygon(box) for box in glb_tile_tree_boxes]}
            
            gdf = gpd.GeoDataFrame(cols, crs = crs)
            #gdf = gpd.GeoDataFrame(geometry=[samplers_utils.xyxyBox2Polygon(box) for box in glb_tile_tree_boxes], crs = crs)
            
            return gdf
        
        return glb_tile_tree_boxes #xyxy format, global index

    def seg_tree_and_build_rnd_samples(self, tile_path, title: str = None, **kwargs):
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
                                                max_area_mt2 = seg_config.max_area_GD_boxes_mt2,
                                                min_edges_ratio = seg_config.min_ratio_GD_boxes_edges,
                                                reduce_perc = seg_config.perc_reduce_tree_boxes)
            
            #BUILDINGS
            #get the building boxes in batches and the number of buildings for each image
            building_boxes_b, num_build4img = detect.get_batch_buildings_boxes(batch['bbox'],
                                                                proj_buildings_gdf = self.proj_build_gdf,
                                                                dataset_res = dataset.res,
                                                                ext_mt = seg_config.ext_mt_build_box)
                        
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
            plot_logits = False
            if not plot_logits:
                patch_masks_b = np.greater_equal(patch_masks_b, 0) #turn logits into bool
                
            
            #plotting
            for img, masks, tree_boxes, building_boxes in zip(img_b, patch_masks_b, tree_boxes_b, building_boxes_b):
                if plot_logits:
                    fig, axs = plt.subplots(1,3,figsize = (20, 20))
                    if title is not None:
                            fig.suptitle(title)
                    axs[0].imshow(masks[1], cmap='hot', interpolation='nearest')
                    plotting_utils.show_box(tree_boxes, axs[0], color='r', lw = 0.4)
                    
                    plotting_utils.show_img(img, ax=axs[1])
                    plotting_utils.show_box(tree_boxes, axs[1], color='r', lw = 0.4)
                    
                    plotting_utils.show_img(img, ax=axs[2])
                    plotting_utils.show_mask(np.greater_equal(masks[1], 0), axs[2], rgb_color = (255, 18, 18), alpha = 0.4)
                    plotting_utils.show_box(tree_boxes, axs[2], color='r', lw = 0.4)
                    
                    return masks[1] 
                    
                    
                else:
                    clean_bool = kwargs.get('clean_bool', False)
                    if clean_bool:
                        masks = segment_utils.clean_masks(masks,
                                                          operations = kwargs.get('operations'),
                                                          area_threshold = kwargs.get('area_threshold'),
                                                          min_size = kwargs.get('min_size')) #clean mask from small particles
                        
                    if True: #plot only trees
                        fig, ax = plt.subplots(figsize = (15, 15))
                        if title is not None:
                                fig.suptitle(title)
                        
                        plotting_utils.show_img(img, ax=ax)
                        plotting_utils.show_mask(masks[0], ax, rgb_color = (255, 18, 18), alpha = 0.4)
                        plotting_utils.show_box(tree_boxes, ax, color='r', lw = 0.4)
                
                    else: #plot trees and buildings 
                        fig, axs = plt.subplots(1, 2, figsize = (16, 8))
                        if title is not None:
                            fig.suptitle(title)
                        #plot trees and build separately
                        plotting_utils.show_img(img, ax=axs[0])
                        plotting_utils.show_mask(masks[0], axs[0], rgb_color = (255, 18, 18), alpha = 0.4)
                        plotting_utils.show_box(tree_boxes, axs[0], color='r', lw = 0.4)
                        
                        plotting_utils.show_img(img, ax = axs[1])
                        plotting_utils.show_mask(masks[1], axs[1], rgb_color = (131, 220, 242), alpha = 0.4)
                        plotting_utils.show_box(building_boxes, axs[1], color='b', lw = 0.4)
    
    def seg_glb_tree_and_build_tile(self, tile_path: str, debug_param_trees_gdf = None, debug = False):
        """
        This method segment trees and buildings of a tile.
        It does compute tree boxes at tile level, NOT at patch level.
        
        Args:
            tile_path: path to the tile to segment
            debug_param_trees_gdf: if not None, the tree boxes are not computed but are taken from this gdf
            debug: if True, the method stops after 50 batches
    
        Returns:
            canvas: a 3 channel mask with trees, buildings and padding. It contains bools
        
        """
        if self.build_gdf is None: #set buildings at mosaic level
            self.set_build_gdf()
        
        if debug_param_trees_gdf is None:
            trees_gdf = self.detect_trees_tile(tile_path, georef = True)
        else:
            trees_gdf = debug_param_trees_gdf
        
        seg_config = self.event.seg_config

        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path))
        sampler = samplers.BatchGridGeoSampler(dataset,
                                            batch_size=seg_config.batch_size,
                                            size=seg_config.size,
                                            stride=seg_config.stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)
        
        canvas = np.full((3,) + samplers_utils.tile_sizes(dataset), fill_value = float(0) ,dtype=np.float32) # dim (3, h_tile, w_tile). The dim 0 is: tree, build, pad
        weights = np.full(samplers_utils.tile_sizes(dataset), fill_value = float(0) ,dtype=np.float32) # dim (h_tile, w_tile)
        
        Esam_total = 0
        post_proc_total = 0
        start_time_all = time()
        for batch_ix, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc = "Segmenting"):
            original_img_tsr = batch['image']

            #TREES 
            #get the tree boxes in batches and the number of trees for each image
            #tree_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di tree boxes
            tree_boxes_b, num_trees4img = detect.get_batch_boxes(batch['bbox'],
                                                                proj_gdf = trees_gdf,
                                                                dataset_res = dataset.res,
                                                                ext_mt = 0)

            #BUILDINGS
            #get the building boxes in batches and the number of buildings for each image
            #building_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di building boxes
            building_boxes_b, num_build4img = detect.get_batch_boxes(batch['bbox'],
                                                                    proj_gdf = self.proj_build_gdf,
                                                                    dataset_res = dataset.res,
                                                                    ext_mt = seg_config.ext_mt_build_box)
            
            
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
            
            canvas, weights = segment_utils.write_canvas_geo_window(canvas = canvas,
                                                                    weights = weights,
                                                                    patch_masks_b =  patch_masks_b,
                                                                    top_lft_indexes = batch['top_lft_index'],
                                                                    )

            # old version
            # canvas = segment_utils.write_canvas_geo(canvas= canvas,
            #                                         patch_masks_b =  patch_masks_b,
            #                                         top_lft_indexes = batch['top_lft_index'],
            #                                         smooth = False)

            post_proc_total += time() - post_proc_start
                
            if batch_ix%100 == 0 and batch_ix != 0:
                print('Avg times (sec/batch)')
                print(f'- ESAM: {(Esam_total/(batch_ix + 1)):.4f}')
                
            #if True and batch_ix == 50:
            #    break
        
        # divide by the weights to get the average
        canvas = np.divide(canvas, weights, out=np.zeros_like(canvas), where=weights!=0)
        canvas = np.greater(canvas, 0) #turn logits into bool
        canvas = np.where(dataset.aoi_mask, canvas, False)
        print(f'\nTotal Time for {seg_config.batch_size * (batch_ix + 1)} images: ', time() - start_time_all)
        return canvas
    
    def new_seg_tree_and_build_tile(self, tile_path):
        """
        This method should segment trees and buildings of a tile
        It should have access to a gdf of trees and a gdf of buildings stored as polygon??
        It should write the segmented mask in a canvas and return it.

        debug = True, if you want to process and plot some #batch_size random samples and plot them
        """
        
        if self.build_gdf is None:
            self.set_build_gdf()

        seg_config = self.event.seg_config
        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path))
        sampler = samplers.BatchGridGeoSampler(dataset, batch_size=seg_config.batch_size, size=seg_config.size, stride=seg_config.stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)

        canvas = np.full((3,) + samplers_utils.tile_sizes(dataset), fill_value = float(0) ,dtype=np.float32) # dim (3, h_tile, w_tile). The dim 0 is: tree, build, pad
        weights = np.full(samplers_utils.tile_sizes(dataset), fill_value = float(0) ,dtype=np.float32) # dim (h_tile, w_tile)

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
                                                max_area_mt2 = seg_config.max_area_GD_boxes_mt2,
                                                min_edges_ratio = seg_config.min_ratio_GD_boxes_edges,
                                                reduce_perc = seg_config.perc_reduce_tree_boxes)
            GD_total += time() - GD_start
            #tree_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di tree boxes
            
            #print('GD_time: ', time() - GD_t_0)

            #get the building boxes in batches and the number of buildings for each image
            build_box_start = time()
            building_boxes_b, num_build4img = detect.get_batch_buildings_boxes(batch['bbox'],
                                                                        proj_buildings_gdf = self.proj_build_gdf,
                                                                        dataset_res = dataset.res,
                                                                        ext_mt = seg_config.ext_mt_build_box)
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
            
            canvas, weights = segment_utils.write_canvas_geo_window(canvas = canvas,
                                                    weights = weights,
                                                    patch_masks_b =  patch_masks_b,
                                                    top_lft_indexes = batch['top_lft_index'],)
             
            # old version
            # canvas = segment_utils.write_canvas_geo(canvas = canvas,
            #                                         patch_masks_b =  patch_masks_b,
            #                                         img_ixs = batch_ix,
            #                                         stride = seg_config.stride)
            

            post_proc_total += time() - post_proc_start
                
            if batch_ix%100 == 0 and batch_ix != 0:
                print('Avg times (sec/batch)')
                print(f'- GD: {(GD_total/(batch_ix + 1)):.4f}')
                print(f'- build_box: {(build_box_total/(batch_ix + 1)):.4f}')
                print(f'- ESAM: {(Esam_total/(batch_ix + 1)):.4f}')
                print(f'- post_proc: {(post_proc_total/(batch_ix + 1)):.4f}')
                
            # if batch_ix == 50:
            #     break
        
        # divide by the weights to get the average
        canvas = np.divide(canvas, weights, out=np.zeros_like(canvas), where=weights!=0)
        canvas = np.greater(canvas, 0) #turn logits into bool
        print(f'\nTotal Time for {seg_config.batch_size * (batch_ix + 1)} images: ', time() - start_time_all)
        return canvas

    def seg_glb_tree_and_build_tile_fast(self, tile_path: str, debug_param_trees_gdf = None, debug = False):
        """
        This method segment trees and buildings of a tile.
        It does compute tree boxes at tile level, NOT at patch level.
        
        Args:
            tile_path: path to the tile to segment
            debug_param_trees_gdf: if not None, the tree boxes are not computed but are taken from this gdf
            debug: if True, the method stops after 50 batches
    
        Returns:
            canvas: a 3 channel mask with trees, buildings and padding. It contains bools
        """
        if self.build_gdf is None: #set buildings at mosaic level
            self.set_build_gdf()
        
        """#TODO:uncomment after debugging-- if debug_param_trees_gdf is None:
            trees_gdf = self.detect_trees_tile(tile_path, georef = True)
        else:
            trees_gdf = debug_param_trees_gdf"""
        
        seg_config = self.event.seg_config

        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path))
        sampler = samplers.BatchGridGeoSampler(dataset,
                                            batch_size=seg_config.batch_size,
                                            size=seg_config.size,
                                            stride=seg_config.stride)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)
        
        canvas = np.full((2,) + samplers_utils.tile_sizes(dataset), fill_value = float(0) ,dtype=np.float32) # dim (3, h_tile, w_tile). The dim 0 is: tree, build, pad
        #weights = np.full(samplers_utils.tile_sizes(dataset), fill_value = float(0) ,dtype=np.float32) # dim (h_tile, w_tile)
        
        for batch_ix, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc = "Segmenting"):

            #TREES 
            #get the tree boxes in batches and the number of trees for each image
            #tree_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di tree boxes
            """#TODO:uncomment after debugging-- tree_boxes_b, num_trees4img = detect.get_batch_boxes(batch['bbox'],
                                                                proj_gdf = trees_gdf,
                                                                dataset_res = dataset.res,
                                                                ext_mt = 0)"""
            
            tree_boxes_b = [np.array([[0,0,0,0]])]
            num_trees4img = np.array([1])
            #BUILDINGS
            #get the building boxes in batches and the number of buildings for each image
            #building_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di building boxes
            building_boxes_b, num_build4img = detect.get_batch_boxes(batch['bbox'],
                                                                    proj_gdf = self.proj_build_gdf,
                                                                    dataset_res = dataset.res,
                                                                    ext_mt = seg_config.ext_mt_build_box)
            
            if num_trees4img[0] > 0 or num_build4img[0] > 0:
                
                original_img_tsr = batch['image']
                max_detect = max(num_trees4img + num_build4img)
                
                #obtain the right input for the ESAM model (trees + buildings)
                input_points, input_labels = segment_utils.get_input_pts_and_lbs(tree_boxes_b, building_boxes_b, max_detect)
                
                # segment the image and get for each image as many masks as the number of boxes,
                # for GPU constraint use num_parall_queries
                
                
                                                        
                tree_build_mask = segment.ESAM_from_inputs_fast(original_img_tsr = original_img_tsr,
                                                            input_points = torch.from_numpy(input_points),
                                                            input_labels = torch.from_numpy(input_labels),
                                                            num_tree_boxes= num_trees4img,
                                                            efficient_sam = seg_config.efficient_sam,
                                                            device = seg_config.device,
                                                            num_parall_queries = 10)
                
                """tree_build_mask = segment.ESAM_from_inputs(original_img_tsr = original_img_tsr,
                                                            input_points = torch.from_numpy(input_points),
                                                            input_labels = torch.from_numpy(input_labels),
                                                            efficient_sam = seg_config.efficient_sam,
                                                            device = seg_config.device,
                                                            num_parall_queries = 2,
                                                            empty_cuda_cache = True)"""
            
            else:
                print('no prompts in patch, skipping...')
                tree_build_mask = np.full((2, *original_img_tsr.shape[2:]), fill_value=float('-inf')) #(2, h, w)
            
            canvas = segment_utils.write_canvas_geo(canvas= canvas,
                                                    patch_masks_b =  np.expand_dims(tree_build_mask, axis=0),
                                                    top_lft_indexes = batch['top_lft_index'],
                                                    smooth = False)

            
            #if batch_ix == 50:
            #    break
            
        canvas = np.greater(canvas, 0) #turn logits into bool
        canvas = np.where(dataset.aoi_mask, canvas, False)
        
        return canvas

    def evit_tree_and_build_tile(self, tile_path, debug = False) -> np.ndarray:
        """
        
        """
        seg_config = self.event.seg_config
        
        if self.build_gdf is None: #set buildings at mosaic level
            self.set_build_gdf()
        
        if self.event.wteSAM is None:
            self.event.load_evit_and_predictor()
        
        #trees_gdf = self.detect_trees_tile(tile_path, georef = True)
        
        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path))
        sampler = samplers.BatchGridGeoSampler(dataset,
                                            batch_size=1,
                                            size=1024,
                                            stride=1024)
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)
        
        #canvas = np.full((2,) + samplers_utils.tile_sizes(dataset), fill_value = float('-inf') ,dtype=np.float32) #dim (2, h_tile, w_tile). The dim 0 is: tree, build
        canvas = np.full((2,) + samplers_utils.tile_sizes(dataset), fill_value = False , dtype= bool)
        weights = np.full(samplers_utils.tile_sizes(dataset), fill_value = float(0), dtype=np.float32) # dim (h_tile, w_tile)

        Esam_total = 0
        post_proc_total = 0
        start_time_all = time()
        
        multimask_output = True #TODO da inserire come parametro
        return_logits = False #TODO da inserire come parametro
        num_parall_queries = 100 #TODO da inserire come parametro
        
        device = self.event.evit_sam_predictor.device
        print('Start segmenting..')
        for batch_ix, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc = "Segmenting"):
            
            original_img_np = batch['image'][0].numpy().transpose(1, 2, 0)
            original_img_h_w = original_img_np.shape[:2] #FIXME: probabliy useless to compute since fixed at the dimension accepted by the model
            patch_bbox = batch['bbox'][0]
            
            #TREES
            """tree_boxes = detect.evit_get_batch_box_prompt(patch_bbox,
                                                        trees_gdf,
                                                        dataset.res,
                                                        ext_mt = 1)"""
            tree_boxes = torch.empty(0, 4)
            #BUILDINGS
            building_boxes = detect.evit_get_batch_box_prompt(patch_bbox,
                                                            self.proj_build_gdf,
                                                            dataset.res,
                                                            ext_mt = 5)
            
            if tree_boxes.shape[0] > 0 or building_boxes.shape[0] > 0:
                boxes = torch.cat((tree_boxes, building_boxes), dim=0).to(device) #no need to apply (rescale) boxes or image since image is enforced to have size == model.image_size
                
                self.event.evit_sam_predictor.set_image(original_img_np, image_format = 'RGB')
                #print('boxes.shape: ', boxes.shape)
                ESAM_start = time()
                
                #SEGMENTING
                tree_build_mask = evit_segment.evit_from_input_maybe_fast(boxes,
                                                    num_tree_boxes = tree_boxes.shape[0],
                                                    evit_sam_predictor = self.event.evit_sam_predictor,
                                                    original_img_h_w = original_img_h_w,
                                                    num_parall_queries = num_parall_queries,
                                                    device = device,
                                                    multimask_output = multimask_output,
                                                    return_logits=return_logits)
                
                
                Esam_total += time() - ESAM_start

            else: 
                #print('no prompts in patch, skipping...')
                if return_logits:
                    tree_build_mask = np.full((2, *original_img_h_w), fill_value=float('-inf')) #(2, h, w)
                else:
                    tree_build_mask = np.full((2, *original_img_h_w), fill_value=False) #(2, h, w)
            
            post_proc_start = time()
            
            if True:
                canvas = segment_utils.write_canvas_geo(canvas= canvas,
                                                        patch_masks_b =  np.expand_dims(tree_build_mask, axis=0),
                                                        top_lft_indexes = batch['top_lft_index'],
                                                        smooth = False)
            
            else:
                canvas, weights = segment_utils.write_canvas_geo_window(canvas = canvas,
                                                                    weights = weights,
                                                                    patch_masks_b = np.expand_dims(tree_build_mask, axis=0),
                                                                    top_lft_indexes = batch['top_lft_index'],
                                                                    )
            #___________________________

            post_proc_total += time() - post_proc_start
                
            #if batch_ix%500 == 0 and batch_ix != 0:
            #    print('Avg times (sec/batch)')
            #    print(f'- ESAM: {(Esam_total/(batch_ix + 1)):.4f}')
            
            if batch_ix == 50:
                break
        
        # divide by the weights to get the average
        if False:
            canvas = np.divide(canvas, weights, out=np.zeros_like(canvas), where=weights!=0)
        
        canvas = np.greater(canvas, 0) #turn logits into bool
        canvas = np.where(dataset.aoi_mask, canvas, False)
        print(f'\nTotal Time for {seg_config.batch_size * (batch_ix + 1)} images: ', time() - start_time_all)
        return canvas
    
    def segment_tile(self, tile_path, out_dir_root, overwrite = False, seg_type: str = 'glbl', separate_masks = True):
        """
        seg_type: if 'glbl', tree detection are computed at tile level (ESAM),
                if 'new', tree are computed at patch level (ESAM),
                if 'evit', tree are computed using 'evit' (ESAM)
        """
        seg_config = self.event.seg_config
        
        if self.build_gdf is None:
            response = self.set_build_gdf()
            if response == False:
                return False
            
        
        tile_path = Path(tile_path)
        out_dir_root = Path(out_dir_root)
                
        out_names = output.gen_names(tile_path, separate_masks)

        (out_dir_root / out_names[0]).parent.mkdir(parents=True, exist_ok=True) #create folder if not exists
        
        if not overwrite:
            for out_name in out_names:
                assert not (out_dir_root / out_name).exists(), f'File {out_name} already exists'
        #############        
        
        tree_and_build_mask = self.seg_glb_tree_and_build_tile_fast(tile_path)
        
        ############
        """if not seg_config.use_evit:
            if seg_type == 'glbl':
                tree_and_build_mask = self.seg_glb_tree_and_build_tile(tile_path)
            elif seg_type == 'new':
                tree_and_build_mask = self.new_seg_tree_and_build_tile(tile_path)
        else:"""
        #tree_and_build_mask = self.evit_tree_and_build_tile(tile_path)
        ####################
        
        
        if self.road_gdf is None:
            self.set_road_gdf()
        road_mask = self.seg_road_tile(tile_path)
        
        overlap_masks = np.concatenate((np.expand_dims(road_mask, axis=0), tree_and_build_mask), axis = 0)
        no_overlap_masks = segment_utils.rmv_mask_overlap(overlap_masks)
        
        if seg_config.clean_masks_bool and False:
            print('Cleaning the masks: holes_area_th = ', seg_config.ski_rmv_holes_area_th, 'small_obj_area = ', seg_config.rmv_small_obj_area_th)
            no_overlap_masks = segment_utils.clean_masks(no_overlap_masks,
                                                        area_threshold = seg_config.ski_rmv_holes_area_th,
                                                        min_size = seg_config.rmv_small_obj_area_th)
        
        output.masks2Tifs(tile_path,
                        no_overlap_masks,
                        out_names = out_names,
                        separate_masks = separate_masks,
                        out_dir_root = out_dir_root)
        return True
        
    def segment_all_tiles(self, out_dir_root, time_per_tile = []):
        for tile_path in self.tiles_paths:
            start_time = perf_counter()
            response = self.segment_tile(tile_path, out_dir_root=out_dir_root, seg_type='evit', separate_masks = False)
            end_time = perf_counter() 
            execution_time = end_time - start_time 
            time_per_tile.append(execution_time)
            print(f'Finished segmenting tile {tile_path} in {execution_time:.2f} seconds')
            print(f'Average time per tile: {np.mean(time_per_tile):.2f} seconds')
            if response == False: #this means that buildings footprint are not available for the mosaic, go to next mosaic
                return time_per_tile, False
        return time_per_tile, True


class Event:
    def __init__(self,
                name,
                seg_config: SegmentConfig = None,
                det_config: DetectConfig = None,
                when = 'pre', #'pre', 'post', None or 'None'
                maxar_root = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data',
                maxar_metadata_path = './metadata/from_github_maxar_metadata/datasets',
                region = 'infer'):
        #Configs
        self.seg_config = seg_config
        self.det_config = det_config
        self.time_per_tile = []
        
        #Paths
        self.maxar_root = Path(maxar_root)
        self.buildings_ds_links_path = Path('./metadata/buildings_dataset_links.csv')
        self.maxar_metadata_path = Path(maxar_metadata_path)
        
        #Models
        self.GD_model = None
        self.DF_model = None
        self.wteSAM = None
        
        #Resizer only for evit
        self.evit_sam_predictor = None
        
        #Event
        self.name = name
        self.when = when
        self.region_name = names.get_region_name(self.name) if region == 'infer' else region
        self.bbox = delimiters.get_event_bbox(self.name, extra_mt=1000) #TODO può essere ottimizzata sfruttando i mosaici
        self.all_mosaics_names = names.get_mosaics_names(self.name, self.maxar_root, self.when)
        
        print(f'Creating event: {self.name}\nRegion: {self.region_name}\nMosaics: {self.all_mosaics_names}')
        #Roads
        self.road_gdf = None

        #Mosaics
        self.mosaics = {}

        #Init mosaics
        for m_name in self.all_mosaics_names:
            self.mosaics[m_name] = Mosaic(m_name, self)

    def set_seg_config(self, seg_config):
        self.seg_config = seg_config
    
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

    def get_roi_polygon(self, wkt: bool = False):
        poly = samplers_utils.xyxy_2_Polygon(self.bbox)
        if wkt:
            return poly.wkt
        else:
            return poly
    
    def get_mosaic(self, mosaic_name):
        return self.mosaics[mosaic_name]

    #Segment methods
    def seg_all_mosaics(self, out_dir_root):
        for __, mosaic in self.mosaics.items():
            times, response = mosaic.segment_all_tiles(out_dir_root=out_dir_root, time_per_tile=self.time_per_tile)
            self.time_per_tile.extend(times)
            if response == False:
                print(f'Buildings footprint not available for mosaic: {mosaic.name}. Proceeding to next mosaic...')
                continue  

    def load_evit_and_predictor(self): #TODO fare tutti i metodi per caricare e scaricare modelli
        self.wteSAM = create_sam_model(name="xl1", weight_url="models/evit/weights/sam/xl1.pt")
        if torch.cuda.is_available():
            self.wteSAM = self.wteSAM.cuda()
        else:
            self.wteSAM = self.wteSAM.cpu()
        self.wteSAM = self.wteSAM.eval()
        
        self.evit_sam_predictor = EfficientViTSamPredictor(self.wteSAM)

    def del_evit(self):
        raise NotImplementedError("Method not implemented")

