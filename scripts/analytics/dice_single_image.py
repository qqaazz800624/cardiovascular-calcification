#%%

import os
import torch
from monai.metrics import DiceMetric
from manafaln.transforms import LoadJSON, OverlayMask, Fill
from deeplabv3plus_custom.parse_labelme import ParseLabelMeDetectionLabel
from manafaln.transforms import ParseXAnnotationSegmentationLabel, Interpolate, Fill, OverlayMask
import matplotlib.pyplot as plt
from monai.transforms import LoadImage, Resize, ScaleIntensity
from heart_seg import HeartSegmentation
import json

def dice_computer(fold_no, img_serial):
    datalist = "/neodata/oxr/tmuh/datalist.json"
    data_root = '../../data/tmuh/'
    datalist = LoadJSON(json_only=True)(datalist)

    #image_file = os.path.join(data_root, datalist[fold_no][img_serial]['image1']) 
    image_loader = LoadImage(image_only=True, ensure_channel_first= True, reader='PydicomReader')
    resizer = Resize(spatial_size = [512, 512])
    scaler = ScaleIntensity()

    model_weight = 'deeplabv3plus_custom/model_ckpts/heart_seg_dropout.ckpt'   
    model_config = {'name': 'DeepLabV3Plus',
                    'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_Dropout',
                    'args':{
                        'in_channels': 3,
                        'classes': 6,
                        'encoder_name': 'tu-resnest50d',
                        'encoder_weights': 'None'}
                        }
    masks_generator = HeartSegmentation(model_config=model_config, model_weight=model_weight, heart_only=True)

    json_file = open('../../data/tmuh/datalist.json')
    img_paths = json.load(json_file)
    path_list = []
    img_no_list = ['image1', 'image2', 'image3']
    for img_no in img_no_list:
        img_tmp = img_paths[fold_no][img_serial][img_no]
        path = os.path.join(data_root, img_tmp)
        path_list.append(path)

    img_pre = []
    for i in range(3):
        img = image_loader(path_list[i])
        img = resizer(img)
        img = scaler(img)
        img_pre.append(img)
    generator_input = torch.cat((img_pre[0], img_pre[1], img_pre[2]), dim=0)
    generator_output = masks_generator(generator_input)
    prediction = torch.from_numpy(generator_output.unsqueeze(0).unsqueeze(0).detach().numpy())

    labelfile = os.path.join(data_root, datalist[fold_no][img_serial]['label'])
    label = LoadJSON(json_only=True)(labelfile)
    label_parser=ParseXAnnotationSegmentationLabel(item_keys= ['#7f007f']) # '#7f007f' the label item for heart
    heart_labels = label_parser(label)
    interpolater = Interpolate(spatial_size = [512, 512])
    interpolated_label = interpolater(heart_labels)
    overlayMasker = OverlayMask(colors=['#7f007f'])
    overlaymask=overlayMasker(image=img_pre[0], masks=interpolated_label[0])
    filler = Fill(mask_idx=[0])
    filled = filler(interpolated_label[0]).astype(int)
    label = torch.from_numpy(filled.reshape((1,) + filled.shape))
    
    return overlaymask, filled, prediction, label, generator_output

## good: 1, 2, 3, 4, 5, 6, 7
## bad: 0

#%%


json_file = open('../../data/tmuh/datalist.json')
img_paths = json.load(json_file)
fold_no='fold_4'
dice_metric = DiceMetric(include_background=True, reduction='mean')
dice_scores = []

from tqdm import tqdm 

for img_serial in tqdm(range(len(img_paths['fold_4']))):
    overlaymask, filled, prediction, label, generator_output = dice_computer(fold_no=fold_no, img_serial=img_serial)
    dice_metric(y_pred=prediction, y=label)
    dice_score = dice_metric.aggregate().item()
    dice_scores.append(dice_score)
    dice_metric.reset()
#%%

import json

# 將列表保存為 JSON 文件
with open('results/dice_scores.json', 'w') as file:
    json.dump(dice_scores, file)

#%%
    

import json

# 從 JSON 文件中讀取列表
with open('results/dice_scores.json', 'r') as file:
    dice_scores = json.load(file)

#%%

import numpy as np

np.array(dice_scores)



#%%
fold_no = 'fold_4'
img_serial = 4
overlaymask, filled, prediction, label, generator_output = dice_computer(fold_no=fold_no, img_serial=img_serial)

#%%

plt.imshow(overlaymask.T, cmap='plasma', aspect='auto')
#%%
plt.imshow(filled.T, cmap='plasma', aspect='auto')

#%%
plt.imshow(generator_output.T, cmap='plasma', aspect='auto')
#%%
# Set include_background to False if you don't want to include the 
#background class (class 0) in the Dice calculation
dice_metric = DiceMetric(include_background=True, reduction='mean')

# Compute Dice score
dice_metric(y_pred=prediction, y=label)
dice_score = dice_metric.aggregate().item()
dice_metric.reset()

print("Dice Score:", dice_score)

#%%



