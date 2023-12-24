#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout
from tqdm import tqdm

def MCDropout(img_no, num_samples):
    
    model_weight = 'deeplabv3plus_custom/model_ckpts/heart_seg_dropout.ckpt'   
    model_config = {'name': 'DeepLabV3Plus',
                    'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_Dropout',
                    #'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_aleatoric',
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


img_no = '022_20221212'
num_samples = 100
output = MCDropout(img_no = img_no, num_samples = num_samples)


def entropy_generator(MCD_output, bins=100, vmin=0, vmax=1, threshold=4):
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    stacked_output = torch.stack(MCD_output, dim=0).to(device)
    entropy_map = torch.zeros(stacked_output.shape[1:], device=device)

    # step 1: compute the entropy map
    for i in tqdm(range(stacked_output.shape[1])):
        for j in range(stacked_output.shape[2]):
            pixel_values = stacked_output[:, i, j]
            histogram = torch.histc(pixel_values, bins=bins, min=vmin, max=vmax)
            p_array = histogram / histogram.sum()
            p_array = p_array.clamp(min=np.finfo(float).eps)  # Avoid division by zero
            entropy_map[i, j] = -torch.sum(p_array * torch.log2(p_array))
    
    # step 2: compute the sum of entropy values greater than the threshold
    entropy_values = entropy_map.view(-1).to('cpu')
    average_entropy = entropy_values[entropy_values > threshold].median()
    return entropy_map.cpu(), average_entropy.numpy()


#%%

# from tqdm import tqdm
# threshold = 0

# for img in tqdm(imgs_list):
#     img_no = img
#     output = MCDropout(img_no = img_no, num_samples = num_samples)
#     entropy_map, image_entropy = entropy_generator(MCD_output = output, threshold=threshold)
#     vmin, vmax = 0, 6
#     plt.imshow(entropy_map.detach().numpy().T, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.xlabel(f'Uncertainty: {image_entropy:.4f}')
#     plt.title(f'Heatmap of Entropy: {img_no}')
#     plt.savefig(f'images/pointwise_entropy_heatmap_{img_no}_v2', bbox_inches='tight')
#     plt.close()

#%%

from tqdm import tqdm
threshold = 0

entropy_map, image_entropy = entropy_generator(MCD_output = output, threshold=threshold)
vmin, vmax = 0, 6
plt.imshow(entropy_map.detach().numpy().T, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel(f'Uncertainty: {image_entropy:.4f}')
plt.title(f'Heatmap of Entropy: {img_no}')
plt.savefig(f'images/pointwise_entropy_heatmap_{img_no}_alea', bbox_inches='tight')
plt.close()

#%%

