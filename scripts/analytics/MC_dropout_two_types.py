#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout

#%%


def MCDropout(num_samples, fold_no, img_serial):

    model_weight = 'deeplabv3plus_custom/model_ckpts/heart_seg_dropout.ckpt'   
    model_config = {'name': 'DeepLabV3Plus',
                    'path': 'seg_model',
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

    data_root = f'../../data/neodata/dicom'
    json_file = open('../../data/neodata/datalist_b_cv.json')
    img_paths = json.load(json_file)

    fold_no = fold_no
    img_serial = img_serial

    path_list = []
    img_no_list = ['image_front_combined', 'image_front_soft', 'image_front_hard']
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


#%%


from monai.transforms import AsDiscrete
from tqdm import tqdm
discreter = AsDiscrete(threshold=0.001)

fold_no_list = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 
                'fold_5', 'fold_6', 'fold_7', 'fold_8', 'fold_9']
fold_size_dict = {'fold_0': 20,
                  'fold_1': 20,
                  'fold_2': 20,
                  'fold_3': 20,
                  'fold_4': 20,
                  'fold_5': 20,
                  'fold_6': 20,
                  'fold_7': 20,
                  'fold_8': 20,
                  'fold_9': 20
                  }
num_samples = 100
uncertainty_list = []

for fold_no in fold_no_list:
    for img_serial in tqdm(range(fold_size_dict[fold_no])):
        output = MCDropout(num_samples, fold_no, img_serial)
        pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)
        uncertainty = discreter(pointwise_variance).sum()
        uncertainty_list.append(uncertainty.detach().numpy().tolist())
        #print(uncertainty_list)

print(uncertainty_list)

#%%

# import json

# # 將列表保存為 JSON 文件
# with open('uncertainty_testing.json', 'w') as file:
#     json.dump(uncertainty_list, file)

#%%

import json

# 從 JSON 文件中讀取列表
with open('uncertainty_testing.json', 'r') as file:
    loaded_list_testing = json.load(file)

with open('uncertainty_list.json', 'r') as file:
    loaded_list_training = json.load(file)
#%%

from matplotlib import pyplot as plt
plt.hist(loaded_list_training, bins=45, color='blue', alpha = 0.9,
         label='Training')
plt.xlim(5000, 45000)
plt.legend()
plt.show()

#%%
from matplotlib import pyplot as plt
plt.hist(loaded_list_testing, bins=45, color='red', alpha = 0.7,
         label='Testing')
plt.xlim(5000, 45000)
plt.legend()
plt.show()

#%%

from matplotlib import pyplot as plt
bins = 45
plt.hist(loaded_list_training, bins=bins, alpha = 0.9, 
         color='blue', label='Training'
         , density=True
         )
plt.hist(loaded_list_testing, bins=bins, alpha = 0.7,
         color='red', label='Testing'
         , density=True
         )
plt.xlim(5000, 45000)
plt.legend()
plt.show()

#%%

# import pandas as pd

# df = pd.DataFrame(loaded_list)
# df.describe()


# import numpy as np

# print(np.sort(loaded_list))

# print(np.quantile(loaded_list, q=np.arange(0.01, 1, 0.01)))


