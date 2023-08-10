#%%
import json

import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage, Resize, ScaleIntensity

datalist_path = "../../data/neodata/datalist.json"

with open(datalist_path, "r") as f:
    datalist = json.load(f)

img_path = '../../data/neodata/dicom/198_20230315/image_front_combined.dcm'

loader = LoadImage(image_only=True, ensure_channel_first= True, reader='PydicomReader')
resizer = Resize(spatial_size = [512, 512])
scaler = ScaleIntensity()
img = loader(img_path)
img = resizer(img)
img = scaler(img)


plt.subplots(1, 3, figsize = (8,8))
plt.subplot(1,3,1)
plt.xlabel('Original')
plt.imshow(img.numpy()[0], cmap='gray')

plt.subplot(1,3,2)
plt.xlabel('Rotated')
plt.imshow(np.rot90(img.numpy()[0], k=3), cmap='gray')

plt.subplot(1,3,3)
plt.xlabel('Transposed')
plt.imshow(img.numpy()[0].T, cmap='gray')
plt.show()

#%%

plt.imshow(img.numpy()[0], cmap='gray')
plt.imshow(np.rot90(img.numpy()[0], k=3), cmap='gray')
plt.imshow(img.numpy()[0].T, cmap='gray')

#%%


