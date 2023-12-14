#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout


def MCDropout(img_no, num_samples):
    
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
        img = loader(path_list[i])
        img = resizer(img)
        img = scaler(img)
        img_pre.append(img)

    generator_input = torch.cat((img_pre[0], img_pre[1], img_pre[2]), dim=0)
    generator_input = generator_input.to(device)
    generator_output = masks_generator(generator_input)
    return generator_output


imgs_list = ['054_20230116', '129_20230216', '144_20230221', '146_20230221',
            '148_20230221', '169_20230306', '198_20230315', '022_20221212',
            '006_20221109', '007_20221109', '010_20221111','012_20221115', 
            '013_20221118', '018_20221206', '025_20221213'
                  ]


img_no = '169_20230306'
num_samples = 100
output = MCDropout(img_no = img_no, num_samples = num_samples)

def entropy_generator(MCD_output):
    stacked_output = torch.stack(MCD_output, dim=0)
    probs = stacked_output / stacked_output.sum(dim=0, keepdim=True)
    entropy_map = -torch.sum(probs * torch.log(probs), dim=0)
    flattened_entropy_map = entropy_map.flatten()
    flattened_p = flattened_entropy_map / flattened_entropy_map.sum()
    image_entropy = -torch.sum(flattened_p * torch.log(flattened_p))
    return entropy_map, np.round(image_entropy.detach().numpy(), 4)

#%%

from tqdm import tqdm

for img in tqdm(imgs_list):
    img_no = img
    output = MCDropout(img_no = img_no, num_samples = num_samples)
    entropy_map, image_entropy = entropy_generator(MCD_output = output)
    plt.imshow(entropy_map.detach().numpy().T, cmap='plasma', aspect='auto')
    plt.colorbar()
    plt.xlabel(f'Shannon Entropy: {image_entropy:.4f}')
    plt.title(f'Heatmap of Entropy: {img_no}')
    plt.savefig(f'images/pointwise_entropy_heatmap_{img_no}', bbox_inches='tight')
    plt.close()

#%%

# import torch

# entropy_map, image_entropy = entropy_generator(MCD_output = output)
# vmin, vmax = 0, 0.03
# plt.imshow(entropy_map.detach().numpy().T, cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.xlabel(f'Shannon Entropy: {image_entropy:.4f}')
# plt.title(f'Heatmap of Entropy: {img_no}')
# plt.savefig(f'images/pointwise_entropy_heatmap_{img_no}', bbox_inches='tight')
# plt.close()


#%%



