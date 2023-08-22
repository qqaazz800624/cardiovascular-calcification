#!/bin/bash

python -m manafaln.utils.cross_validation \
    -c ${2:-"configs/2_cac_classification/train_b2.yaml"}\
    -k ${3:-5}\
    -e \
    -v ${1:-3} 

