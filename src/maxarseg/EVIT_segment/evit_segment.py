import sys
from cv2 import threshold
import torch
import numpy as np
from maxarseg.EVIT_segment import evit_utils
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

def evit_from_input_old(boxes: torch.Tensor, num_tree_boxes: int,
                    evit_sam_predictor: EfficientViTSamPredictor,
                    original_img_h_w: tuple,
                    num_parall_queries: int,
                    device: torch.device,
                    multimask_output = True,
                    return_logits = True) -> np.ndarray:
    
    device = evit_sam_predictor.device
    
    if multimask_output == True:
        all_masks = np.empty((0, 3, *original_img_h_w),
                            dtype=np.float32)
        all_quality = np.empty((0, 3),
                                dtype=np.float32)
    else:
        all_masks = np.empty((0, 1, *original_img_h_w),
                                dtype=np.float32)
        all_quality = np.empty((0, 1),
                                dtype=np.float32)         

    for i in range(0, boxes.shape[0], num_parall_queries):
        batch_boxes = boxes[i:i+num_parall_queries]
        print('boxes_area: ', torch.sum((batch_boxes[:,2]-batch_boxes[:,0])*(batch_boxes[:,3]-batch_boxes[:,1])).item())
        try: #FIXME: cuda out of memory error. strange since sometimes 200 boxes are not a problem, but sometimes with less boxes it breaks
            print('boxes[i:i+num_parall_queries].shape: ', batch_boxes.shape)
            masks, quality, low_res_logits = evit_sam_predictor.predict_torch(boxes = boxes[i:i+num_parall_queries],
                                                                                multimask_output=multimask_output,
                                                                                return_logits=return_logits) # shape: (n_boxes, n_masks, h_patch, w_patch)
            
            all_masks = np.concatenate((all_masks, masks.cpu().numpy()), axis=0)
            all_quality = np.concatenate((all_quality, quality.cpu().numpy()), axis=0)
        
        except Exception as e:
            torch.cuda.memory._dump_snapshot(f"my_snapshot_ERROR_np.pickle")
            print('\n\n#####ERROR#####\n\n')
            print(e)
            print('boxes.shape: ', boxes.shape)
            print('boxes[i:i+num_parall_queries].shape: ', boxes[i:i+num_parall_queries].shape)
            print('i: ', i)
            print('all_masks.shape: ', all_masks.shape)
            print('all_quality.shape: ', all_quality.shape)
            sys.exit()

    #for each box take mask with best quality
    best_masks = all_masks[np.arange(all_masks.shape[0]), np.argmax(all_quality, axis=1)]
    tree_build_mask = evit_utils.separate_classes_numpy(best_masks, num_trees = num_tree_boxes)
    
    
    return tree_build_mask

def evit_from_input_maybe_fast(boxes: torch.Tensor,
                                num_tree_boxes: int,
                                evit_sam_predictor: EfficientViTSamPredictor,
                                original_img_h_w: tuple,
                                num_parall_queries: int,
                                device: torch.device,
                                multimask_output = True,
                                return_logits = True) -> np.ndarray:
    
    device = evit_sam_predictor.device
    
    tree_build_mask = torch.full((2, *original_img_h_w), float('-inf'), dtype = torch.float32, device = device)
    num_batch_tree_only = num_tree_boxes // num_parall_queries
    trees_in_mixed_batch = round(num_parall_queries * (num_tree_boxes/num_parall_queries -  num_tree_boxes // num_parall_queries))

    for y, i in enumerate(range(0, boxes.shape[0], num_parall_queries)):
        batch_boxes = boxes[i:i+num_parall_queries]
        #print('boxes_area: ', torch.sum((batch_boxes[:,2]-batch_boxes[:,0])*(batch_boxes[:,3]-batch_boxes[:,1])).item())
        #print('boxes[i:i+num_parall_queries].shape: ', batch_boxes.shape)
        masks, quality, low_res_logits = evit_sam_predictor.predict_torch(boxes = boxes[i:i+num_parall_queries],
                                                                            multimask_output=multimask_output,
                                                                            return_logits=return_logits) # shape: (n_boxes, n_masks, h_patch, w_patch)
        #get the best masks
        masks = masks[torch.arange(masks.shape[0]), torch.argmax(quality, dim=1)] #shape (n_boxes, h_patch, w_patch)
        
        del low_res_logits
        del quality
        
        if y < num_batch_tree_only or boxes[i:i+num_parall_queries].shape[0] == trees_in_mixed_batch: #only trees
            tree_build_mask[0] = torch.max(tree_build_mask[0], torch.max(masks, dim=0).values)
        elif y > num_batch_tree_only or trees_in_mixed_batch == 0: #only build
            tree_build_mask[1] = torch.max(tree_build_mask[1], torch.max(masks, dim=0).values)
        else: #trees and build
            tree_build_mask[0] = torch.max(tree_build_mask[0], torch.max(masks[:trees_in_mixed_batch], dim=0).values)
            tree_build_mask[1] = torch.max(tree_build_mask[1], torch.max(masks[trees_in_mixed_batch:], dim=0).values)
        
        del masks
        
    tree_build_mask = tree_build_mask.cpu().numpy()
    return tree_build_mask

def evit_from_input_x_boxes_cpu(boxes: torch.Tensor, num_tree_boxes: int,
                        evit_sam_predictor: EfficientViTSamPredictor,
                        original_img_h_w: tuple,
                        num_parall_queries: int,
                        device: torch.device,
                        multimask_output = True,
                        return_logits = True) -> np.ndarray:
    
    device = evit_sam_predictor.device
    
    #tmp_all_masks = evit_utils.placeholder_masks(original_img_h_w, device, 'torch')
    tmp_all_masks = torch.empty((0, *original_img_h_w), dtype = torch.float32, device = device)
    #all_masks_np = evit_utils.placeholder_masks(original_img_h_w, device, 'numpy')
    all_masks_np = np.empty((0, *original_img_h_w), dtype=np.float32)
    
    #masks_on_gpu = 0
    threshold_to_cpu = 400 #TODO: aggingere come parametro
    for i in range(0, boxes.shape[0], num_parall_queries):
        batch_boxes = boxes[i:i+num_parall_queries]
        print('boxes_area: ', torch.sum((batch_boxes[:,2]-batch_boxes[:,0])*(batch_boxes[:,3]-batch_boxes[:,1])).item())

        print('boxes[i:i+num_parall_queries].shape: ', batch_boxes.shape)
        masks, quality, low_res_logits = evit_sam_predictor.predict_torch(boxes = boxes[i:i+num_parall_queries],
                                                                            multimask_output=multimask_output,
                                                                            return_logits=return_logits) # shape: (n_boxes, n_masks, h_patch, w_patch)
        
        #get the best masks
        masks = masks[torch.arange(masks.shape[0]), torch.argmax(quality, dim=1)] #shape (n_boxes, h_patch, w_patch)
        
        print('masks_shape', masks.shape)
        tmp_all_masks = torch.cat((tmp_all_masks, masks), dim=0)
        
        #masks_on_gpu += num_parall_queries
        masks_on_gpu = tmp_all_masks.shape[0]
        
        if masks_on_gpu >= threshold_to_cpu or (i + num_parall_queries) >= boxes.shape[0]:
            print('\nMoving masks to gpu. Num. mask_on_gpu: ', masks_on_gpu)
            tmp_all_masks = tmp_all_masks.cpu().numpy() #move to cpu
            all_masks_np = np.concatenate((all_masks_np, tmp_all_masks), axis=0)
            tmp_all_masks = evit_utils.placeholder_masks(original_img_h_w, device, 'torch')
                


    #for each box take mask with best quality
    tree_build_mask = evit_utils.separate_classes_numpy(all_masks_np, num_trees = num_tree_boxes)
    return tree_build_mask