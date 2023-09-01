#%%

import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from seg_model import DeepLabV3Plus


#%%

import torch
from manafaln.core.builders import ModelBuilder

model_weight = 'heart_seg.ckpt'
model_config = {'name': 'DropoutDeepLabV3Plus',
                'path': 'DropoutDeepLabV3Plus',
                'args':{
                    'in_channels': 3,
                    'classes': 6,
                    'encoder_name': 'tu-resnest50d',
                    'encoder_weights': 'None'}
                }

model = ModelBuilder()(model_config)

#%%
model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(model_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    model_weight[k_new] = model_weight.pop(k)

model.load_state_dict(model_weight)


#%%


model = DeepLabV3Plus(in_channels=3, classes = 6, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')

model.eval()


img_no = '006_20221109'
img1_path = f'../../data/neodata/dicom/{img_no}/image_front_combined.dcm'
img2_path = f'../../data/neodata/dicom/{img_no}/image_front_soft.dcm'
img3_path = f'../../data/neodata/dicom/{img_no}/image_front_hard.dcm'

loader = LoadImage(image_only=True, ensure_channel_first= True, reader='PydicomReader')
resizer = Resize(spatial_size = [512, 512])
scaler = ScaleIntensity()


img1 = loader(img1_path)
img1 = resizer(img1)
img1 = scaler(img1)

img2 = loader(img2_path)
img2 = resizer(img2)
img2 = scaler(img2)


img3 = loader(img3_path)
img3 = resizer(img3)
img3 = scaler(img3)

img = torch.cat((img1, img2, img3), dim=0)

img = img.unsqueeze(0)


model.random_forward(img)


#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout


def MCDropout(img_no, num_samples):
    
    model_weight = 'heart_seg_dropout.ckpt'
    model_config = {'name': 'DeepLabV3Plus',
                    'path': 'seg_model',
                    'args':{
                        'in_channels': 3,
                        'classes': 6,
                        'encoder_name': 'tu-resnest50d',
                        'encoder_weights': 'None'}
                    }
    
    img_no = img_no
    num_samples = num_samples

    loader = LoadImage(image_only=True, ensure_channel_first= True, reader='PydicomReader')
    resizer = Resize(spatial_size = [512, 512])
    scaler = ScaleIntensity()
    masks_generator = SegmentationMCDropout(model_config=model_config, model_weight=model_weight, num_samples=num_samples)

    data_root = f'../../data/neodata/dicom/{img_no}'
    path_list = []
    img_typ = ['image_front_combined.dcm', 'image_front_soft.dcm', 'image_front_hard.dcm']
    for typ in img_typ:
        path = os.path.join(data_root, typ)
        path_list.append(path)

    img_pre = []
    for i in range(3):
        img = loader(path_list[0])
        img = resizer(img)
        img = scaler(img)
        img_pre.append(img)

    generator_input = torch.cat((img_pre[0], img_pre[1], img_pre[2]), dim=0)
    generator_output = masks_generator(generator_input)
    return generator_output

    
output = MCDropout(img_no = '054_20230116', num_samples = 10)


#%%

output[0]


#%%

output[1]

#%%

img_no = '054_20230116'
for i in range(10):

    plt.subplots(1, 3, figsize = (8,8))

    plt.subplot(1,3,1)
    plt.xlabel('Original')
    plt.imshow(output[i].numpy(), cmap='gray')

    plt.subplot(1,3,2)
    plt.title(f'{img_no}')
    plt.xlabel('Rotated')
    plt.imshow(np.rot90(output[i].numpy(), k=3), cmap='gray')

    plt.subplot(1,3,3)
    plt.xlabel('Transposed')
    plt.imshow(output[i].numpy().T, cmap='gray')

    plt.show()


#%%