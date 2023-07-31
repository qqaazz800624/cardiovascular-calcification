python -m manafaln.apps.validate \
    -c ${2:-"configs/1_heart_segmentation/train_b.yaml"} \
    -f lightning_logs/version_5/fold_$1/checkpoints/best_model.ckpt
    #-f lightning_logs/$1/checkpoints/best_model.ckpt