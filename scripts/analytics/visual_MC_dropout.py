#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from scripts.analytics.heart_seg_MCDropout import HeartSegmentationMCDropout


def visual_single_img(img_no):
    
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

    loader = LoadImage(image_only=True, ensure_channel_first= True, reader='PydicomReader')
    resizer = Resize(spatial_size = [512, 512])
    scaler = ScaleIntensity()
    model = HeartSegmentationMCDropout(model_config=model_config, model_weight=model_weight, heart_only=True)

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

    model_input = torch.cat((img_pre[0], img_pre[1], img_pre[2]), dim=0)
    model_output = model(model_input)

    plt.subplots(1, 3, figsize = (8,8))

    plt.subplot(1,3,1)
    plt.xlabel('Original')
    plt.imshow(model_output.numpy(), cmap='gray')

    plt.subplot(1,3,2)
    plt.title(f'{img_no}')
    plt.xlabel('Rotated')
    plt.imshow(np.rot90(model_output.numpy(), k=3), cmap='gray')

    plt.subplot(1,3,3)
    plt.xlabel('Transposed')
    plt.imshow(model_output.numpy().T, cmap='gray')

    plt.show()

img_list = ['054_20230116', '129_20230216', '144_20230221', '146_20230221',
            '148_20230221', '169_20230306', '198_20230315']

for img_no in img_list:
    visual_single_img(img_no) 


#%%