#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout


def MCDropout(img_no, num_samples):
    
    model_weight = 'heart_seg.ckpt'    
    model_config = {'name': 'DeepLabV3Plus',
                    'path': 'seg_model',
                    'args':{
                        'in_channels': 3,
                        'classes': 6,
                        'encoder_name': 'tu-resnest50d',
                        'encoder_weights': 'None'}
                    }
    
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
        img = loader(path_list[0])
        img = resizer(img)
        img = scaler(img)
        img_pre.append(img)

    generator_input = torch.cat((img_pre[0], img_pre[1], img_pre[2]), dim=0)
    generator_output = masks_generator(generator_input)
    return generator_output


# bad_imgs_list = ['054_20230116', '129_20230216', '144_20230221', '146_20230221',
#                  '148_20230221', '169_20230306', '198_20230315', '022_20221212']
# good_imgs_list = ['006_20221109', '007_20221109', '010_20221111','012_20221115', 
#                   '013_20221118', '018_20221206', '025_20221213']
imgs_list = ['054_20230116', '129_20230216', '144_20230221', '146_20230221',
            '148_20230221', '169_20230306', '198_20230315', '022_20221212',
            '006_20221109', '007_20221109', '010_20221111','012_20221115', 
            '013_20221118', '018_20221206', '025_20221213'
                  ]
#img_no = bad_imgs_list[5] #bad examples
#img_no = good_imgs_list[0] #good examples
img_no = '007_20221109'
num_samples = 10
output = MCDropout(img_no = img_no, num_samples = num_samples)


#%%
from monai.transforms import AsDiscrete
discreter = AsDiscrete(threshold=0.001)

for img in imgs_list:
    img_no = img
    output = MCDropout(img_no = img_no, num_samples = num_samples)
    pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)
    high_var = discreter(pointwise_variance).sum()
    vmin, vmax = 0, 0.03
    plt.imshow(pointwise_variance.detach().numpy().T, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel(f'Uncertainty: {high_var}')
    plt.title(f'Heatmap of Variance: {img_no}')
    plt.savefig(f'images/pointwise_var_heatmap_{img_no}', bbox_inches='tight')
    plt.close()

#%%
from monai.transforms import AsDiscrete
discreter = AsDiscrete(threshold=0.2)

for i in range(10):
    #high_logits = discreter(output[i]).sum()
    plt.imshow(output[i].detach().numpy().T, cmap='plasma', aspect='auto')
    plt.colorbar()
    #plt.xlabel(f'The number of high logits: {high_logits}')
    plt.title(f'Heatmap of sample_{i}')
    plt.show()


#%%
# from monai.transforms import AsDiscrete
# discreter = AsDiscrete(threshold=0.001)
# pointwise_variance = torch.stack(output, dim=0).var(dim=0, keepdim=False)
# high_var = discreter(pointwise_variance).sum()
# vmin, vmax = 0, 0.03
# plt.imshow(pointwise_variance.detach().numpy().T, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.xlabel(f'Uncertainty: {high_var}')
# plt.title(f'Heatmap of Variance: {img_no}')
# plt.savefig(f'images/pointwise_var_heatmap_{img_no}', bbox_inches='tight')
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