#%%
import torch
import torch.nn as nn
from manafaln.core.builders import ModelBuilder
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2

#%%

class efficientnet_ensemble(nn.Module):
    def __init__(
        self, 
        num_classes: int,
    ):
        super().__init__()
        self.model1 = efficientnet_b0(weights='DEFAULT')
        self.model2 = efficientnet_b1(weights='DEFAULT')
        #self.model3 = efficientnet_b2(weights='DEFAULT')
        self.linear = nn.Linear(1000, num_classes)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        #x3 = self.model3(x)
        x_combined = (x1 + x2)/2
        #x_combined = (x1 + x2 + x3)/3
        x_output = self.linear(x_combined)
        return x_output

