#!/bin/bash

python -m manafaln.apps.validate \
    -c ${2:-"configs/2_cac_classification/test_b.yaml"} \
    -f lightning_logs/version_2/fold_9/checkpoints/best_model.ckpt
    #-f lightning_logs/$1/checkpoints/best_model.ckpt
