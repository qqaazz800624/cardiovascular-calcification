#%%

import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from heart_seg_MCDropout import HeartSegmentationMCDropout
#from segmentation_models_pytorch import DeepLabV3Plus
from seg_model import DeepLabV3Plus


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