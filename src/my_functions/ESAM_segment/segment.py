import numpy as np
import torch


def ESAM_from_inputs(original_img_tsr: torch.tensor, #b, c, h, w
                    input_points: torch.tensor, #b, max_queries, 2, 2
                    input_labels: torch.tensor, #b, max_queries, 2
                    efficient_sam,
                    num_parall_queries: int = 50,
                    device = 'cpu',
                    empty_cuda_cache = True):
    
    img_b_tsr = original_img_tsr.div(255)
    batch_size, _, input_h, input_w = img_b_tsr.shape
    
    img_b_tsr = img_b_tsr.to(device)
    input_points = input_points.to(device)
    input_labels = input_labels.to(device)

    image_embeddings = efficient_sam.get_image_embeddings(img_b_tsr)
    
    stop = input_points.shape[1]
    if stop > 0: #if there is at least a query in a single image in the batch
        for i in range(0, stop , num_parall_queries):
            start_idx = i
            end_idx = min(i + num_parall_queries, stop)
            #TODO: check if multimask_output False is faster
            predicted_logits, predicted_iou = efficient_sam.predict_masks(image_embeddings,
                                                                    input_points[:, start_idx: end_idx],
                                                                    input_labels[:, start_idx: end_idx],
                                                                    multimask_output=True,
                                                                    input_h = input_h,
                                                                    input_w = input_w,
                                                                    output_h=input_h,
                                                                    output_w=input_w)
            
            if i == 0:
                #print('predicetd_logits:', predicted_logits.shape)
                np_complete_masks = predicted_logits[:,:,0].cpu().detach().numpy()
            else:
                np_complete_masks = np.concatenate((np_complete_masks, predicted_logits[:,:,0].cpu().detach().numpy()), axis=1)
            #TODO: check if empty_cuda_cache Fasle is faster
            if empty_cuda_cache:
                del predicted_logits, predicted_iou
                torch.cuda.empty_cache()
    else: #if there are no queries (in any image in the batch)
        np_complete_masks = np.ones((batch_size, 0, input_h, input_w)) * -1 #equal to set False on all the mask
        
    
    return np_complete_masks #shape (b, masks, h, w)