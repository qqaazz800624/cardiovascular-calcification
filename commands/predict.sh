#!/bin/bash

python -m manafaln.apps.predict \
    -c ${1:-"configs/1_heart_segmentation/predict_b.yaml"} \
    -f custom/transforms/heart_seg_dropout.ckpt
    #-f lightning_logs/$1/checkpoints/best_model.ckpt
