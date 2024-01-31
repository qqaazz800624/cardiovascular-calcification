#%%

import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout
from tqdm import tqdm 

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

#%%

num_samples = 100
fold_no = 'fold_4'
img_serial = 0
output = MCDropout(num_samples, fold_no, img_serial)

#%%

stacked_tensor = torch.stack(output, dim=0)
eta  = torch.exp(stacked_tensor)/(1+torch.exp(stacked_tensor))
stacked_eta = eta*(1-eta)
first_term  = stacked_eta.mean(dim = 0, keepdim=False)
taylor_estimate = (stacked_tensor+2)/4
second_term = taylor_estimate.var(dim = 0, keepdim=False)
epistemic_uncertainty = first_term + second_term
aleatoric_uncertainty = stacked_tensor.var(dim = 0, keepdim=False)/16

plt.imshow(aleatoric_uncertainty.detach().numpy().T, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Heatmap of Aleatoric Uncertainty: {fold_no}_{img_serial}')

#%%

plt.imshow(epistemic_uncertainty.detach().numpy().T, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Heatmap of Epistemic Uncertainty: {fold_no}_{img_serial}')

#%%

total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
plt.imshow(total_uncertainty.detach().numpy().T, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Heatmap of Total Uncertainty: {fold_no}_{img_serial}')

#%%
def uncertainty_estimator(T, num_samples, fold_no, img_serial):
    means_list = []
    squared_means_list = []
    vars_list = []
    for t in tqdm(range(T)):
        output = MCDropout(num_samples, fold_no, img_serial)
        sample_means = torch.stack(output, dim=0).mean(dim=0, keepdim=False)
        sample_vars = torch.stack(output, dim=0).var(dim=0, keepdim=False)
        squared_means = sample_means**2
        means_list.append(sample_means)
        squared_means_list.append(squared_means)
        vars_list.append(sample_vars)
    
    first_term = torch.stack(squared_means_list, dim = 0).mean(dim=0, keepdim=False)
    second_term = torch.stack(means_list, dim = 0).mean(dim=0, keepdim=False)**2
    epistemic_uncertainty = first_term - second_term
    aleatoric_uncertainty = torch.stack(vars_list, dim = 0).mean(dim=0, keepdim=False)
    


    #%%
    plt.imshow(epistemic_uncertainty.detach().numpy().T, cmap='plasma', aspect='auto', vmin = 0, vmax = 0.03)
    plt.colorbar()
    plt.title(f'Heatmap of Epistemic: {fold_no}_{img_serial}')
    plt.savefig(f'images/epistemic_heatmap_{fold_no}_{img_serial}', bbox_inches='tight')
    plt.close()

    plt.imshow(aleatoric_uncertainty.detach().numpy().T, cmap='plasma', aspect='auto', vmin = 0, vmax = 0.03)
    plt.colorbar()
    plt.title(f'Heatmap of Aleatoric: {fold_no}_{img_serial}')
    plt.savefig(f'images/aleatoric_heatmap_{fold_no}_{img_serial}', bbox_inches='tight')
    plt.close()


T = 1000
num_samples = 1000
fold_no = 'fold_4'
img_serial_list = [0, 2, 4, 41]

for img_serial in tqdm(img_serial_list):
    uncertainty_estimator(T, num_samples, fold_no, img_serial)

#%%

# T = 10
# num_samples = 10
# fold_no = 'fold_4'
# img_serial =  4

# means_list = []
# squared_means_list = []
# vars_list = []
# for t in range(T):
#     output = MCDropout(num_samples, fold_no, img_serial)
#     sample_means = torch.stack(output, dim=0).mean(dim=0, keepdim=False)
#     sample_vars = torch.stack(output, dim=0).var(dim=0, keepdim=False)
#     squared_means = sample_means**2
#     means_list.append(sample_means)
#     squared_means_list.append(squared_means)
#     vars_list.append(sample_vars)

# first_term = torch.stack(squared_means_list, dim = 0).mean(dim=0, keepdim=False)
# second_term = torch.stack(means_list, dim = 0).mean(dim=0, keepdim=False)**2
# epistemic_uncertainty = first_term - second_term
# aleatoric_uncertainty = torch.stack(vars_list, dim = 0).mean(dim=0, keepdim=False)

# plt.imshow(epistemic_uncertainty.detach().numpy().T, cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Heatmap of Epistemic: {fold_no}_{img_serial}')
# plt.savefig(f'images/epistemic_heatmap_{fold_no}_{img_serial}', bbox_inches='tight')
# plt.close()

# plt.imshow(aleatoric_uncertainty.detach().numpy().T, cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Heatmap of Aleatoric: {fold_no}_{img_serial}')
# plt.savefig(f'images/aleatoric_heatmap_{fold_no}_{img_serial}', bbox_inches='tight')
# plt.close()

#%%




#%%





#%%




#%%





#%%




#%%