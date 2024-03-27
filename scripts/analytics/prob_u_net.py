#%%

from segmentation_models_pytorch import Unet

import os
import tempfile
from functools import partial

import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from lightning_uq_box.uq_methods.prob_unet import ProbUNet

from tmuh_datamodule import TMUHDataModule
#%%

my_temp_dir = 'results/'

network = Unet(in_channels=3, classes=2, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
Prob_UNet = ProbUNet(
    model=network,
    optimizer=partial(torch.optim.AdamW, lr=3.0e-4, weight_decay=1e-5, amsgrad=True)
    )

data_module = TMUHDataModule()

logger = CSVLogger(my_temp_dir)
trainer = Trainer(
    accelerator='gpu',
    devices=2,
    max_epochs=32,  # number of epochs we want to train
    logger=logger,  # log training metrics for later evaluation
    log_every_n_steps=8,
    enable_checkpointing=True,
    enable_progress_bar=True,
    default_root_dir=my_temp_dir,
    num_sanity_val_steps=2
)

trainer.fit(Prob_UNet, data_module)

#%%






#%%






#%%






#%%





#%%






#%%






#%%







#%%





#%%






#%%






#%%