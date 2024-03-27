#!/bin/bash
python /home/vaschetti/maxarSrc/src/main_seg_tile.py \
        --event_ix 6 \
        --when pre \
        --bs 2 \
        --size 700 \
        --stride 700 \
        --device cuda:3 \
        --text_prompt "green tree" \
        --box_threshold 0.12 \
        --text_threshold 0.30 \
        --max_area_GD_boxes_mt2 6000 \
        --min_ratio_GD_boxes_edges 0.7 \
        --perc_reduce_tree_boxes 0.2 \
        --road_width_mt 5 \
        --ext_mt_build_box 0 \
        --ESAM_num_parall_queries 5 \
        --out_dir_root "/home/vaschetti/maxarSrc/output/tiff"