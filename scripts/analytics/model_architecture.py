#%%

import torch
from manafaln.core.builders import ModelBuilder

model_weight = 'deeplabv3plus_custom/model_ckpts/heart_seg_dropout.ckpt'
model_config = {'name': 'DeepLabV3Plus',
                #'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_Dropout',
                'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_aleatoric',
                'args':{
                    'in_channels': 3,
                    'classes': 6,
                    'encoder_name': 'tu-resnest50d',
                    'encoder_weights': 'None'}
                }

model = ModelBuilder()(model_config)

model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(model_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    model_weight[k_new] = model_weight.pop(k)

model.load_state_dict(model_weight)

module_list = [module for module in model.modules()]
children_list = [child for child in model.children()]
#children_list
#print(children_list)
print(module_list)

#%%


