#%%
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToySegmentationDataset
from tmuh_dataset import TMUHDataset

from monai.transforms import Compose,LoadImaged, Resized, ScaleIntensityd
from monai.transforms import ConcatItemsd, RandAffined, DeleteItemsd, EnsureTyped
from manafaln.transforms import LoadJSONd, ParseXAnnotationSegmentationLabeld, Interpolated, Filld, OverlayMaskd

class TMUHDataModule(LightningDataModule):
    """Toy segmentation datamodule."""

    def __init__(self, batch_size=16):
        """Initialize a toy image segmentation datamodule.

        Args:
            num_images: number of images in the dataset
            image_size: size of the image
            batch_size: batch size
        """
        super().__init__()
        self.batch_size = batch_size
        
        self.train_folds = ['fold_0', 'fold_1', 'fold_2']
        self.val_folds = ['fold_3']
        self.test_folds = ['fold_4']

        self.train_transforms = Compose([
                                LoadImaged(keys=['image1', 'image2', 'image3'], 
                                        ensure_channel_first=True,
                                        reader='PydicomReader',
                                        image_only=True),
                                Resized(keys=['image1', 'image2', 'image3'], 
                                        spatial_size=[512, 512]),
                                ScaleIntensityd(keys=['image1', 'image2', 'image3']),
                                ConcatItemsd(keys=['image1', 'image2', 'image3'], name='image'),
                                LoadJSONd(keys='target'), 
                                ParseXAnnotationSegmentationLabeld(keys='target',
                                                                item_keys=['#ff7f00', '#ff0000', '#7f007f', '#0000ff', '#00ff00', '#ffff00']),
                                Interpolated(keys='target', spatial_size=[512, 512]),
                                DeleteItemsd(keys='target_meta_dict.meta_points'),
                                OverlayMaskd(image_keys='image1', 
                                            mask_keys='target', 
                                            names='target_visual', 
                                            colors=['#ff7f00', '#ff0000', '#7f007f', 
                                                    '#0000ff', '#00ff00', '#ffff00']),
                                Filld(keys='target'),
                                RandAffined(keys=['image', 'target'],
                                            prob=1.0,
                                            rotate_range=0.25,
                                            shear_range=0.2,
                                            translate_range=0.1,
                                            scale_range=0.2,
                                            padding_mode='zeros'),
                                EnsureTyped(keys=['image', 'target', 'target_meta_dict'], dtype='float32')
                                ])
        
        self.val_transforms =   Compose([
                                LoadImaged(keys=['image1', 'image2', 'image3'], 
                                        ensure_channel_first=True,
                                        reader='PydicomReader',
                                        image_only=True),
                                Resized(keys=['image1', 'image2', 'image3'], 
                                        spatial_size=[512, 512]),
                                ScaleIntensityd(keys=['image1', 'image2', 'image3']),
                                ConcatItemsd(keys=['image1', 'image2', 'image3'], name='image'),
                                LoadJSONd(keys='target'), 
                                ParseXAnnotationSegmentationLabeld(keys='target',
                                                                item_keys=['#ff7f00', '#ff0000', '#7f007f', '#0000ff', '#00ff00', '#ffff00']),
                                Interpolated(keys='target', spatial_size=[512, 512]),
                                DeleteItemsd(keys='target_meta_dict.meta_points'),
                                OverlayMaskd(image_keys='image1', 
                                            mask_keys='target', 
                                            names='target_visual', 
                                            colors=['#ff7f00', '#ff0000', '#7f007f', 
                                                    '#0000ff', '#00ff00', '#ffff00']),
                                Filld(keys='target'),
                                EnsureTyped(keys=['image', 'target', 'target_meta_dict'], dtype='float32')
                                ])
        
        self.test_transforms = Compose([
                                LoadImaged(keys=['image1', 'image2', 'image3'], 
                                        ensure_channel_first=True,
                                        reader='PydicomReader',
                                        image_only=True),
                                Resized(keys=['image1', 'image2', 'image3'], 
                                        spatial_size=[512, 512]),
                                ScaleIntensityd(keys=['image1', 'image2', 'image3']),
                                ConcatItemsd(keys=['image1', 'image2', 'image3'], name='image'),
                                LoadJSONd(keys='target'), 
                                ParseXAnnotationSegmentationLabeld(keys='target',
                                                                item_keys=['#ff7f00', '#ff0000', '#7f007f', '#0000ff', '#00ff00', '#ffff00']),
                                Interpolated(keys='target', spatial_size=[512, 512]),
                                DeleteItemsd(keys='target_meta_dict.meta_points'),
                                Filld(keys='target'),
                                EnsureTyped(keys=['image', 'target', 'target_meta_dict'], dtype='float32')
                                ])

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            TMUHDataset(folds=self.train_folds, transform=self.train_transforms),
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the val dataloader."""
        return DataLoader(
            TMUHDataset(folds=self.val_folds, transform=self.val_transforms),
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            TMUHDataset(folds=self.test_folds, transform=self.test_transforms),
            batch_size=self.batch_size,
        )


#%%