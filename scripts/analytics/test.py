#%%

from deeplabv3plus_custom.models.DeepLabV3Plus_alea_entropy import DeepLabV3Plus


#%%

import torch
from manafaln.core.builders import ModelBuilder

model_weight = 'deeplabv3plus_custom/model_ckpts/heart_seg_dropout.ckpt'
model_config = {'name': 'DeepLabV3Plus',
                #'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_Dropout',
                #'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_aleatoric',
                'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_alea_entropy',
                'args':{
                    'in_channels': 3,
                    'classes': 6,
                    'encoder_name': 'tu-resnest50d',
                    'encoder_weights': 'None'}
                }

model = ModelBuilder()(model_config)


#%%





#%%




#%%




#%%