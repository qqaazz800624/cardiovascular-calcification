#%%

import torch
from AddGaussianNoiseModule import AddNoise
from models.DeepLabV3Plus_Dropout import DeepLabV3Plus

#%%
model_weight = 'model_ckpts/heart_seg_dropout.ckpt'  
model =  DeepLabV3Plus(in_channels=3, classes=6, encoder_name='tu-resnest50d', encoder_weights=None)
model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]

for k in list(model_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    model_weight[k_new] = model_weight.pop(k)

model.load_state_dict(model_weight)
#%%

len([param for param in model.parameters()])

#%%
weights_dict = model.state_dict()
weight_keys_list = list(weights_dict.keys())


#%%





#%%




