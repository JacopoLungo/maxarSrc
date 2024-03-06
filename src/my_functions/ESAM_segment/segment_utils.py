import numpy as np
from typing import List, Tuple, Union

def get_input_pts_and_lbs(tree_boxes_b: List, #list of array of shape (query_img_x, 4)
                          building_boxes_b: List, 
                          max_detect: int):
    input_lbs = []
    input_pts = []
    pts_pad_value = -10
    lbs_pad_value = 0
    for tree_detec, build_detec in zip(tree_boxes_b, building_boxes_b):
        tree_build_detect = np.concatenate((tree_detec, build_detec)) #(query_img_x, 4)
        lbs = np.array([[2,3]]*tree_build_detect.shape[0]) #(query_img_x, 2)

        pad_len = max_detect - (tree_build_detect.shape[0] + 1)
        print(pad_len)
        pad_width = ((0,pad_len),(0, 0))
        padded_tree_build_detect = np.pad(tree_build_detect, pad_width, constant_values=pts_pad_value)
        img_input_pts = np.expand_dims(padded_tree_build_detect, axis = 0).reshape(-1,2,2) # (max_queries, 2, 2)
        input_pts.append(img_input_pts)

        padded_lbs = np.pad(lbs, pad_width, constant_values = lbs_pad_value)# (max_queries, 2)
        input_lbs.append(padded_lbs)

    return np.array(input_pts), np.array(input_lbs) # (batch_size, max_queries, 2, 2), (batch_size, max_queries, 2)

def discern(all_mask_b: np.array, num_trees4img:np.array, num_build4img: np.array):
    h, w = all_mask_b.shape[2:]
    tree_mask_b = build_mask_b = pad_mask_b = np.full((1, h, w), False)

    all_mask_b = np.greater_equal(all_mask_b, 0) #from logits to bool
    for all_mask, tree_ix, build_ix in zip (all_mask_b, num_trees4img, num_build4img):
        #all_mask.shape = (num_mask, h, w)
        tree_mask = all_mask[ : tree_ix].any(axis=0) #(h, w)
        tree_mask_b = np.concatenate((tree_mask_b, tree_mask[None, ...]), axis=0) #(b, h, w)

        build_mask = all_mask[tree_ix : (tree_ix + build_ix)].any(axis=0)
        build_mask_b = np.concatenate((build_mask_b, build_mask[None, ...]), axis=0)
        
        pad_mask = all_mask[(tree_ix + build_ix) : ].any(axis=0)
        pad_mask_b = np.concatenate((pad_mask_b, pad_mask[None, ...]), axis=0)
    
    return tree_mask_b[1:], build_mask_b[1:], pad_mask_b[1:] #all (b, h, w), slice out the first element


def rmv_mask_b_overlap(overlapping_masks_b: np.array): #(b, c, h, w)
    """
    Remove overlapping between the masks. Giving priority according to the inverse of the order of
    the masks.
    Third (building) mask has priority over second (trees) mask, and so on.
    """

    disjoined_masks_b = np.copy(overlapping_masks_b)
    for i in range(overlapping_masks_b.shape[1] - 1):
        sum_mask = np.sum(overlapping_masks_b[:,i:], axis=1)
        disjoined_masks_b[:,i] = np.where(sum_mask > 1, False, overlapping_masks_b[:, i])
    
    return disjoined_masks_b

#Use the batch version of this function
def rmv_mask_overlap(overlapping_masks: np.array):
    """
    Remove overlapping between the masks. Giving priority according to the inverse of the order of
    the masks.
    Third (building) mask has priority over second (trees) mask, and so on.
    """
    disjoined_masks = np.copy(overlapping_masks)
    for i in range(overlapping_masks.shape[0] - 1):
        sum_mask = np.sum(overlapping_masks[i:], axis=0)
        disjoined_masks[i] = np.where(sum_mask > 1, False, overlapping_masks[i])

    return disjoined_masks