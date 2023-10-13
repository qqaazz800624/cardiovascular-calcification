#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout
from segmentation_MCDropout_alea import SegmentationMCDropout_alea


def MCDropout(num_samples, fold_no, img_serial):

    model_weight = 'deeplabv3plus_custom/model_ckpts/heart_seg_dropout.ckpt' 
    
    model_config = {'name': 'DeepLabV3Plus',
                    'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_Dropout',
                    'args':{
                        'in_channels': 3,
                        'classes': 6,
                        'encoder_name': 'tu-resnest50d',
                        'encoder_weights': 'None'}
                    }

    device = "cuda:3" if torch.cuda.is_available() else "cuda:2"
    num_samples = num_samples

    loader = LoadImage(image_only=True, ensure_channel_first= True, reader='PydicomReader')
    resizer = Resize(spatial_size = [512, 512])
    scaler = ScaleIntensity()
    masks_generator = SegmentationMCDropout(model_config=model_config, model_weight=model_weight, num_samples=num_samples)

    data_root = f'../../data/tmuh/'
    json_file = open('../../data/tmuh/datalist.json')
    img_paths = json.load(json_file)

    fold_no = fold_no
    img_serial = img_serial

    path_list = []
    img_no_list = ['image1', 'image2', 'image3']
    for img_no in img_no_list:
        img_tmp = img_paths[fold_no][img_serial][img_no]
        path = os.path.join(data_root, img_tmp)
        path_list.append(path)

    img_pre = []
    for i in range(3):
        img = loader(path_list[i])
        img = resizer(img)
        img = scaler(img)
        img_pre.append(img)

    generator_input = torch.cat((img_pre[0], img_pre[1], img_pre[2]), dim=0)
    generator_input = generator_input.to(device)
    generator_output = masks_generator(generator_input)
    return generator_output


def compute_img_entropy(image_array, bins = 6, vmin = 0, vmax = 0.03):
    image_array = image_array
    bin_len = (vmax - vmin)/bins
    thresholds = np.round(np.arange(vmin, vmax, bin_len), 4).tolist()
    thresholds.append(vmax)
    histogram, bins = np.histogram(image_array, bins=thresholds)
    pixel_total = image_array.shape[0]*image_array.shape[1]
    p_array = histogram/pixel_total
    # Shannon Entropy
    entropy = -np.sum(p_array * np.log2(p_array + np.finfo(float).eps))
    return entropy

#%%


from monai.transforms import AsDiscrete
from tqdm import tqdm
discreter = AsDiscrete(threshold=0.001)

fold_no_list = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
fold_size_dict = {'fold_0': 122,
                  'fold_1': 122,
                  'fold_2': 122,
                  'fold_3': 122,
                  'fold_4': 121
                  }
num_samples = 100
uncertainty_list = []

for fold_no in fold_no_list:
    for img_serial in tqdm(range(fold_size_dict[fold_no])):
        output = MCDropout(num_samples, fold_no, img_serial)
        pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)
        image_array = pointwise_variance.detach().numpy()
        uncertainty = compute_img_entropy(image_array)
        uncertainty_list.append(uncertainty)
        #print(uncertainty_list)

print(uncertainty_list)

#%%

import json

# 將列表保存為 JSON 文件
with open('results/uncertainties_train_entropy.json', 'w') as file:
    json.dump(uncertainty_list, file)

#%%

import json

# 從 JSON 文件中讀取列表
with open('results/uncertainties_train_entropy.json', 'r') as file:
    loaded_list = json.load(file)

#%%

from matplotlib import pyplot as plt
#plt.hist(loaded_list, bins='auto')
plt.hist(loaded_list, bins=30, color='blue')
plt.xlim(0.05, 0.5)
plt.show()

#%%

import pandas as pd

df = pd.DataFrame(loaded_list)
df.describe()

#%%
import numpy as np

print(np.sort(loaded_list))
#%%
print(np.quantile(loaded_list, q=np.arange(0.01, 1, 0.01)))


