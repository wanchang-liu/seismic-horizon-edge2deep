import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(
            x))  # Compress global spatial information via average pooling: (B,C,H,W) --> (B,C,1,1), then use MLP to reduce and increase dimensions: (B,C,1,1)
        max_out = self.mlp(self.max_pool(
            x))  # Compress global spatial information via max pooling: (B,C,H,W) --> (B,C,1,1), then use MLP to reduce and increase dimensions: (B,C,1,1)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1,
                             keepdim=True)  # Compress global channel information via average pooling: (B,C,H,W) --> (B,1,H,W)
        max_out, _ = torch.max(x, dim=1,
                               keepdim=True)  # Compress global channel information via max pooling: (B,C,H,W) --> (B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)  # Concatenate the two matrices along the channel dimension: (B,2,H,W)
        x = self.conv(x)  # Get the attention weights via convolution: (B,2,H,W) --> (B,1,H,W)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(
            x)  # Feature map obtained by channel attention mechanism, x:(B,C,H,W), ca(x):(B,C,1,1), out:(B,C,H,W)
        x = x * self.sa(
            x)  # Feature map obtained by spatial attention mechanism, x:(B,C,H,W), sa(x):(B,1,H,W), out:(B,C,H,W)
        return x
