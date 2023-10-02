#!/bin/bash

python -m manafaln.apps.train \
    -s 42 \
    -c ${1:-"configs/1_heart_segmentation/train_b_dropout.yaml"}
