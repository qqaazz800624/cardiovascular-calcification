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

# num_samples = 10
# fold_no = 'fold_0'
# img_serial =  2
# output = MCDropout(num_samples, fold_no, img_serial)

# from monai.transforms import AsDiscrete
# discreter = AsDiscrete(threshold=0.001)
# pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)
# high_var = discreter(pointwise_variance).sum()

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
        uncertainty = discreter(pointwise_variance).sum()
        uncertainty_list.append(uncertainty.detach().numpy().tolist())
        #print(uncertainty_list)

print(uncertainty_list)

#%%

import json

# 將列表保存為 JSON 文件
with open('uncertainty_list.json', 'w') as file:
    json.dump(uncertainty_list, file)

#%%

import json

# 從 JSON 文件中讀取列表
with open('uncertainty_list.json', 'r') as file:
    loaded_list = json.load(file)

#%%

from matplotlib import pyplot as plt
#plt.hist(loaded_list, bins='auto')
plt.hist(loaded_list, bins=45, color='blue')
plt.xlim(5000, 45000)
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



#%%

# for img_serial in range(122):
#     output = MCDropout(num_samples, fold_no, img_serial)
#     pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)
#     uncertainty = discreter(pointwise_variance).sum()
#     uncertainty_list.append(uncertainty)




#%%
# from monai.transforms import AsDiscrete
# discreter = AsDiscrete(threshold=0.01)

# for img in imgs_list:
#     img_no = img
#     output = MCDropout(img_no = img_no, num_samples = num_samples)
#     pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)
#     high_var = discreter(pointwise_variance).sum()
#     vmin, vmax = 0, 0.05
#     plt.imshow(pointwise_variance.detach().numpy().T, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.xlabel(f'Uncertainty: {high_var}')
#     plt.title(f'Heatmap of Variance: {img_no}')
#     plt.savefig(f'images/pointwise_var_heatmap_{img_no}', bbox_inches='tight')
#     plt.close()

#%%

# for i in range(10):
#     vmin, vmax = 0, 0.6
#     plt.imshow(output[i].detach().numpy().T, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.title(f'Heatmap of sample_{i}')
#     plt.show()


#%%
# from monai.transforms import AsDiscrete
# discreter = AsDiscrete(threshold=0.001)
# pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)
# high_var = discreter(pointwise_variance).sum()
# vmin, vmax = 0, 0.03
# plt.imshow(pointwise_variance.detach().numpy().T, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.xlabel(f'Uncertainty: {high_var}')
# plt.title(f'Heatmap of Variance: {fold_no}_{img_serial}')
# plt.savefig(f'images/pointwise_var_heatmap_{fold_no}_{img_serial}', bbox_inches='tight')
# plt.close()

# plt.imshow(pointwise_variance.detach().numpy().T, cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Heatmap of Variance: {img_no}')
# plt.savefig(f'images/pointwise_var_heatmap_{img_no}', bbox_inches='tight')
# plt.close()


#%%

# pointwise_mean = torch.stack(output, dim=0).mean(dim=0, keepdim=False)

# plt.imshow(pointwise_mean.detach().numpy().T, cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Heatmap of Posterior Mean: {img_no}')
# plt.savefig(f'images/posterior_mean_heatmap_{img_no}', bbox_inches='tight')
# plt.close()



#%%