#%%
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity
from segmentation_MCDropout import SegmentationMCDropout
from segmentation_MCDropout_alea import SegmentationMCDropout_alea

#%%
import torch

img_no = '022_20221212'
#img_no = '006_20221109'
heatmap = []
type_list = ['epistemic', 'aleatoric']
for uncertainty_typ in type_list:
    load_path = f'results/heatmap_{uncertainty_typ}_{img_no}.pth'
    heatmap.append(torch.load(load_path))


#%%


def compute_img_entropy(image_array, bins = 6, vmin = 0, vmax = 0.03):
    image_array = image_array
    bin_len = (vmax - vmin)/bins
    thresholds = np.round(np.arange(vmin, vmax, bin_len), 4).tolist()
    thresholds.append(vmax)
    histogram, bins = np.histogram(image_array, bins=thresholds)
    pixel_total = heatmap[sample_no].shape[0]*heatmap[sample_no].shape[1]
    p_array = histogram/pixel_total
    # Shannon Entropy
    entropy = -np.sum(p_array * np.log2(p_array + np.finfo(float).eps))
    return entropy, image_array


#%%

# Overall Uncertainty: epistemic + aleatoric uncertainty
import matplotlib.pyplot as plt

sample_no = 0
image_array = heatmap[sample_no].detach().numpy()
entropy, image_array = compute_img_entropy(image_array)
vmin, vmax = 0, 0.03
plt.imshow(image_array, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel(f'Shannon Entropy: {np.round(entropy, 4)}')
plt.title(f'Overall Uncertainty: {img_no}')
plt.show()

#%% 
# aleatoric uncertainty

sample_no = 1
image_array = heatmap[sample_no].detach().numpy()
entropy, image_array = compute_img_entropy(image_array)
vmin, vmax = 0, 0.03
plt.imshow(image_array, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel(f'Shannon Entropy: {np.round(entropy, 4)}')
plt.title(f'Aleatoric Uncertainty: {img_no}')
plt.show()

#%% 
# epistemic uncertainty obtained by Richardson-Lucy Deconvolution

from skimage import restoration

img0 = heatmap[0].detach().numpy()
img1 = heatmap[1].detach().numpy()
deconvolved_RL = restoration.richardson_lucy(image = img0, psf = img1, num_iter=30)

entropy, image_array = compute_img_entropy(deconvolved_RL)
vmin, vmax = 0, 0.03
plt.imshow(image_array, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel(f'Shannon Entropy: {np.round(entropy, 4)}')
plt.title(f'Epistemic Uncertainty: {img_no}')
plt.show()


#%%





#%%