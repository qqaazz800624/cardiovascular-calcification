#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from heart_seg_MCDropout import HeartSegmentationMCDropout
from seg_model import DeepLabV3Plus

model_weight = 'heart_seg_dropout.ckpt'
model_config = {'name': 'DeepLabV3Plus',
                'path': 'seg_model',
                'args':{
                    'in_channels': 3,
                    'classes': 6,
                    'encoder_name': 'tu-resnest50d',
                    'encoder_weights': 'None'}
                }

img_no = '006_20221109'
img1_path = f'../../data/neodata/dicom/{img_no}/image_front_combined.dcm'
img2_path = f'../../data/neodata/dicom/{img_no}/image_front_soft.dcm'
img3_path = f'../../data/neodata/dicom/{img_no}/image_front_hard.dcm'

loader = LoadImage(image_only=True, ensure_channel_first= True, reader='PydicomReader')
resizer = Resize(spatial_size = [512, 512])
scaler = ScaleIntensity()
model = HeartSegmentationMCDropout(model_config=model_config, model_weight=model_weight, heart_only=True)


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
#%%

img.shape


#%%

img = model(img)


#%%

plt.subplots(1, 3, figsize = (8,8))

plt.subplot(1,3,1)
plt.xlabel('Original')
plt.imshow(img.numpy(), cmap='gray')

plt.subplot(1,3,2)
plt.title(f'{img_no}')
plt.xlabel('Rotated')
plt.imshow(np.rot90(img.numpy(), k=3), cmap='gray')

plt.subplot(1,3,3)
plt.xlabel('Transposed')
plt.imshow(img.numpy().T, cmap='gray')

plt.show()



#%%

import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from heart_seg import HeartSegmentation

def visual_single_img(img_no):
    
    model_weight = 'heart_seg_dropout.ckpt'
    model_config = {'name': 'DeepLabV3Plus',
                    'path': 'segmentation_models_pytorch',
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
    model = HeartSegmentation(model_config=model_config, model_weight=model_weight, heart_only=True)

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

import json

import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity

def visual_single_img_tmuh(file_no, img_no):
    
    img_path = f'../../data/tmuh/image/{file_no}/{img_no}.dcm'

    loader = LoadImage(image_only=True, ensure_channel_first= True, reader='PydicomReader')
    resizer = Resize(spatial_size = [512, 512])
    scaler = ScaleIntensity()
    img = loader(img_path)
    img = resizer(img)
    img = scaler(img)

    plt.subplots(1, 3, figsize = (8,8))

    plt.subplot(1,3,1)
    plt.xlabel('Original')
    plt.imshow(img.numpy()[0], cmap='gray')

    plt.subplot(1,3,2)
    plt.title(f'{img_no}')
    plt.xlabel('Rotated')
    plt.imshow(np.rot90(img.numpy()[0], k=3), cmap='gray')

    plt.subplot(1,3,3)
    plt.xlabel('Transposed')
    plt.imshow(img.numpy()[0].T, cmap='gray')

    plt.show()

visual_single_img_tmuh(file_no='17d7f24f6ae', img_no='018463ee')

#%%


