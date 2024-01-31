#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout
from scripts.analytics.segmentation_MCDropout import SegmentationMCDropout

#%%
import torch

#img_no = '022_20221212'
img_no = '006_20221109'
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

from skimage import restoration

img0 = heatmap[0].detach().numpy()
img1 = heatmap[1].detach().numpy()

# normalization
# img0 = img0/np.sum(img0)
# img1 = img1/np.sum(img1)

deconvolved_RL = restoration.richardson_lucy(image = img0, psf = img1, num_iter=30)

#%%

vmin, vmax = 0, 0.03
plt.imshow(deconvolved_RL, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title(f'Uncertainty by RL-Deconvolution: {img_no}')
plt.show()

#%%





#%%