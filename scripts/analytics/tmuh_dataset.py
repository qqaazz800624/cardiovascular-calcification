#%%

import json
from typing import Dict, List
from torch.utils.data import Dataset
import torch
from torch import Tensor
import os

class TMUHDataset(Dataset):
    """Image segmentation dataset based on a JSON data list."""

    def __init__(self, 
                 folds: List[str], 
                 transform=None, 
                 json_path: str = '/neodata/oxr/tmuh/datalist_b.json',
                 data_root: str = '/neodata/oxr/tmuh/'):
        """Initialize the dataset.

        Args:
            json_path: Path to the JSON file containing data paths.
            folds: List of fold names to use (e.g., ['fold_0', 'fold_1', 'fold_2'] for training).
            transform: Optional transform to be applied on a sample.
            data_root: Root directory of the dataset.
        """
        with open(json_path) as f:
            self.data_list = json.load(f)

        self.transform = transform
        self.data_root = data_root
        self.samples = []
        for fold in folds:
            self.samples.extend(self.data_list[fold])

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.samples[idx]
    
        # Concatenate the data_root with the relative paths
        image1_path = os.path.join(self.data_root, sample['image1'])
        image2_path = os.path.join(self.data_root, sample['image2'])
        image3_path = os.path.join(self.data_root, sample['image3'])
        target_path = os.path.join(self.data_root, sample['target'])

        data_list = {
            'image1': image1_path, 
            'image2': image2_path, 
            'image3': image3_path, 
            'target': target_path
        }

        # Apply transforms if any
        if self.transform:
            transformed = self.transform(data_list)
            image = transformed['image']  # Assuming your transform concatenates and returns 'image'
            target = transformed['target'][2]  # 2: the heart margin mask

        return {'input': image, 'target': target}


#%%


# from torchvision import transforms
# import monai

# from monai.transforms import Compose,LoadImaged, Resized, ScaleIntensityd
# from monai.transforms import ConcatItemsd, RandAffined, DeleteItemsd, EnsureTyped
# from manafaln.transforms import LoadJSONd, ParseXAnnotationSegmentationLabeld, Interpolated, Filld, OverlayMaskd

# train_folds = ['fold_0', 'fold_1', 'fold_2']
# val_folds = ['fold_3']
# test_folds = ['fold_4']


# # Example transforms for training dataset
# train_transforms = Compose([
#     # MONAI transforms for medical images or torchvision transforms for natural images
#     # Add random transformations only to the training set
#     LoadImaged(keys=['image1', 'image2', 'image3'], 
#                ensure_channel_first=True,
#                reader='PydicomReader',
#                image_only=True),
#     Resized(keys=['image1', 'image2', 'image3'], 
#             spatial_size=[512, 512]),
#     ScaleIntensityd(keys=['image1', 'image2', 'image3']),
#     ConcatItemsd(keys=['image1', 'image2', 'image3'], name='image'),
#     LoadJSONd(keys='target'), 
#     ParseXAnnotationSegmentationLabeld(keys='target',
#                                        item_keys=['#ff7f00', '#ff0000', '#7f007f', '#0000ff', '#00ff00', '#ffff00']),
#     Interpolated(keys='target', spatial_size=[512, 512]),
#     DeleteItemsd(keys='target_meta_dict.meta_points'),
#     OverlayMaskd(image_keys='image1', 
#                  mask_keys='target', 
#                  names='target_visual', 
#                  colors=['#ff7f00', '#ff0000', '#7f007f', 
#                          '#0000ff', '#00ff00', '#ffff00']),
#     Filld(keys='target'),
#     RandAffined(keys=['image', 'target'],
#                 prob=1.0,
#                 rotate_range=0.25,
#                 shear_range=0.2,
#                 translate_range=0.1,
#                 scale_range=0.2,
#                 padding_mode='zeros'),
#     EnsureTyped(keys=['image', 'target', 'target_meta_dict'], dtype='float32')
# ])

# # Example transforms for validation dataset
# val_transforms = Compose([
#     # Usually, the validation set only goes through deterministic preprocessing
#     LoadImaged(keys=['image1', 'image2', 'image3'], 
#                ensure_channel_first=True,
#                reader='PydicomReader',
#                image_only=True),
#     Resized(keys=['image1', 'image2', 'image3'], 
#             spatial_size=[512, 512]),
#     ScaleIntensityd(keys=['image1', 'image2', 'image3']),
#     ConcatItemsd(keys=['image1', 'image2', 'image3'], name='image'),
#     LoadJSONd(keys='target'), 
#     ParseXAnnotationSegmentationLabeld(keys='target',
#                                        item_keys=['#ff7f00', '#ff0000', '#7f007f', '#0000ff', '#00ff00', '#ffff00']),
#     Interpolated(keys='target', spatial_size=[512, 512]),
#     DeleteItemsd(keys='target_meta_dict.meta_points'),
#     OverlayMaskd(image_keys='image1', 
#                  mask_keys='target', 
#                  names='target_visual', 
#                  colors=['#ff7f00', '#ff0000', '#7f007f', 
#                          '#0000ff', '#00ff00', '#ffff00']),
#     Filld(keys='target'),
#     EnsureTyped(keys=['image', 'target', 'target_meta_dict'], dtype='float32')
# ])

# # Now, apply these transforms to your datasets when you instantiate them

# train_dataset = TMUHDataset(folds=train_folds, transform=train_transforms)
# val_dataset = TMUHDataset(folds=val_folds, transform=val_transforms)

#%%
