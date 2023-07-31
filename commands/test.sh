#!/bin/bash

python -m manafaln.apps.validate \
    -c ${1:-"configs/2_cac_classification/test_b2.yaml"} \ 
    -f lightning_logs/version_3/fold_5/checkpoints/best_model.ckpt
    #-f lightning_logs/$1/checkpoints/best_model.ckpt
