import torch
import torch.nn as nn
from torch import Tensor

class AddNoise(nn.Sequential):
    """
    Add Gaussian noise to the input tensor.

    Args:
        noise_stddev (float): Standard deviation of Gaussian noise to add. Default is 1.
    """
    __constants__ = ['noise_stddev']
    noise_stddev: float

    def __init__(self, noise_stddev: float = 1):
        super(AddNoise, self).__init__()
        self.noise_stddev = noise_stddev

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply noise to the input tensor.

        Args:
            input (Tensor): Input tensor to which noise will be added.

        Returns:
            Tensor: Output tensor with added noise.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if input.dtype not in (torch.float32, torch.float64):
            raise ValueError("Input tensor must have float32 or float64 data type.")
        if self.noise_stddev <= 0:
            raise ValueError("Noise standard deviation must be a positive value.")

        noise = torch.randn_like(input) * self.noise_stddev
        output = input + noise
        return output
