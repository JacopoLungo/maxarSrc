#!/bin/bash
python ./src/main_seg_tile.py \
        --event_ix 6 \
        --when pre \
        --bs 2 \
        --device cuda:1 \
        --size 600 \
        --stride 600 \
        --text_prompt "green tree" \
        --box_threshold 0.15 \
        --text_threshold 0.30 \
        --max_area_GD_boxes_mt2 6000 \
        --road_width_mt 5 \
        --ESAM_num_parall_queries 5 \
        --out_dir_root "./output/tiff"


