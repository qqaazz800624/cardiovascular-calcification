import torch
from monai.transforms import Transform, MaskIntensity, AsDiscrete

from manafaln.core.builders import ModelBuilder


class SegmentationMCDropout(Transform):
    def __init__(
        self, model_config: dict, model_weight: str, num_samples: int = 10
    ):
        self.num_samples = num_samples

        self.model: torch.nn.Module = ModelBuilder()(model_config)

        model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
        for k in list(model_weight.keys()):
            k_new = k.replace(
                "model.", "", 1
            )  # e.g. "model.conv.weight" => conv.weight"
            model_weight[k_new] = model_weight.pop(k)

        self.model.load_state_dict(model_weight)
        self.model.eval()

    def __call__(self, data):
        # img shape (C, W, H) => (B, C, W, H)
        img = data.unsqueeze(0)
        #discreter = AsDiscrete(threshold=0.5)
        sample_masks = []
        for _ in range(self.num_samples):
            logit = self.model.random_forward(img)
            #mask_heart = logit[0, 2]                    # take segmentation mask of heart
            mask_heart = torch.sigmoid(logit)[0, 2]    # take segmentation mask of heart
            #mask_heart = discreter(mask_heart)
            sample_masks.append(mask_heart)
        
        return sample_masks