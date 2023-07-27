python -m manafaln.apps.validate \
    -c ${1:-"configs/1_heart_segmentation/train_b.yaml"} \ 
    -f lightning_logs/version_4/fold_0/checkpoints/best_model.ckpt
    #-f lightning_logs/$1/checkpoints/best_model.ckpt