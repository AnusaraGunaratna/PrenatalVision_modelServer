import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    def __init__(self, c1, reduction=32):
        super().__init__()
        self.c1 = c1
        mid = max(8, c1 // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Collapse width
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Collapse height

        self.conv1 = nn.Conv2d(c1, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mid, c1, 1)  # Height attention projection
        self.conv_w = nn.Conv2d(mid, c1, 1)  # Width attention projection

    def forward(self, x):
        B, C, H, W = x.shape

        # Encode spatial info in two 1D directions
        x_h = self.pool_h(x)                              # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)          # [B, C, W, 1]

        # Concatenate along spatial dim and transform
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))

        # Split back into height and width attention maps
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Apply directional attention
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))
