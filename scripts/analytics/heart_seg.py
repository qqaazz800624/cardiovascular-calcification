import torch
from monai.transforms import Transform, MaskIntensity, AsDiscrete

from manafaln.core.builders import ModelBuilder


class HeartSegmentation(Transform):
    def __init__(
        self, model_config: dict, model_weight: str, heart_only: bool = False
    ):
        self.heart_only = heart_only

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
        logit = self.model(img)
        discreter = AsDiscrete(threshold=0.5)
        mask_heart = torch.sigmoid(logit)[0, 2]   # take segmentation mask of heart
        if self.heart_only:
            output = discreter(mask_heart)
        else:
            mask_heart = mask_heart.unsqueeze(0)
            maskintensity = MaskIntensity(mask_data=mask_heart)
            output = maskintensity(img.squeeze(0))
        return output
