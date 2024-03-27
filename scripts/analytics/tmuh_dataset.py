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



#%%
