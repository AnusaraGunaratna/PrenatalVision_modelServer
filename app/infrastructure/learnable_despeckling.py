import torch
import torch.nn as nn
import torch.nn.functional as F

# Learnable Despeckling Block (LDB)

import torch
import torch.nn as nn

class LearnableDespeckling(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2

        # Branch 1 - estimates local mean 
        self.mean_conv = nn.Conv2d(
            c1, c1, kernel_size, padding=pad, groups=c1, bias=False
        )
        # Branch 2 - estimates local mean of squares 
        self.sq_conv = nn.Conv2d(
            c1, c1, kernel_size, padding=pad, groups=c1, bias=False
        )
        self.mask_conv = nn.Conv2d(c1, c1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean   = self.mean_conv(x)
        mean_sq = self.sq_conv(x * x)
        variance = (mean_sq - mean * mean).clamp(min=0)
        noise_mask = self.sigmoid(self.mask_conv(variance))
        return x * (1.0 - noise_mask)                

print('LDB defined')