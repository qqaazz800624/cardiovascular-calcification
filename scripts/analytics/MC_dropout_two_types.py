#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout
from scripts.analytics.segmentation_MCDropout_no_sigmoid import SegmentationMCDropout

def MCDropout(img_no, num_samples, uncertainty_typ = 'epistemic'):
    
    model_weight = 'deeplabv3plus_custom/model_ckpts/heart_seg_dropout.ckpt' 

    if uncertainty_typ == 'epistemic': 
        model_config = {'name': 'DeepLabV3Plus',
                        'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_Dropout',
                        'args':{
                            'in_channels': 3,
                            'classes': 6,
                            'encoder_name': 'tu-resnest50d',
                            'encoder_weights': 'None'}
                        }
        
    elif uncertainty_typ == 'aleatoric':
        model_config = {'name': 'DeepLabV3Plus',
                        'path': 'deeplabv3plus_custom.models.DeepLabV3Plus_aleatoric',
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
    
    if uncertainty_typ == 'epistemic':
        masks_generator = SegmentationMCDropout(model_config=model_config, model_weight=model_weight, num_samples=num_samples)
    elif uncertainty_typ == 'aleatoric':
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
            '148_20230221', '169_20230306', '198_20230315', '022_20221212','234_20230328',

            '006_20221109', '007_20221109', '010_20221111','012_20221115', 
            '013_20221118', '018_20221206', '025_20221213'
                  ]

img_no = '144_20230221'
num_samples = 100
uncertainty_typ = 'epistemic'
#uncertainty_typ = 'aleatoric'
output = MCDropout(img_no = img_no, num_samples = num_samples, uncertainty_typ = uncertainty_typ)
pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)

heatmap_output = pointwise_variance.T
save_path = f'results/heatmap_{uncertainty_typ}_{img_no}.pth'
torch.save(heatmap_output, save_path)

#%%
import torch

img_no = '144_20230221'
heatmap = []
type_list = ['epistemic', 'aleatoric']
for uncertainty_typ in type_list:
    load_path = f'results/heatmap_{uncertainty_typ}_{img_no}.pth'
    heatmap.append(torch.load(load_path))

#%%
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import AsDiscrete

discreter = AsDiscrete(threshold=0.01)

reduced_heatmap = heatmap[0] - heatmap[1]
high_var = discreter(reduced_heatmap).sum()
vmin, vmax = 0, 0.03
plt.imshow(reduced_heatmap.detach().numpy(), cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel(f'Uncertainty: {high_var}')
plt.title(f'Epistemic Uncertainty: {img_no}')
plt.show()

#%%

vmin, vmax = 0, 0.03
high_var = discreter(heatmap[0]).sum()
plt.imshow(heatmap[0].detach().numpy(), cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel(f'Uncertainty: {high_var}')
plt.title(f'Uncertainty: {img_no}')
plt.show()

#%%

vmin, vmax = 0, 0.03
high_var = discreter(heatmap[1]).sum()
plt.imshow(heatmap[1].detach().numpy(), cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel(f'Uncertainty: {high_var}')
plt.title(f'Aleatoric Uncertainty: {img_no}')
plt.show()


#%%