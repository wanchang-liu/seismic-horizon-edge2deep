from torch import nn
from models.segnet.segnet_parts import ConvBNReLUMaxPol, ConvBNReLU, Conv


class SegNetBasic(nn.Module):
    def __init__(self, n_channels, n_classes):
        # Initialization
        super().__init__()

        # Encoder
        self.encode1 = ConvBNReLUMaxPol(n_channels, 64)
        self.encode2 = ConvBNReLUMaxPol(64, 128)
        self.encode3 = ConvBNReLUMaxPol(128, 256)
        self.encode4 = ConvBNReLUMaxPol(256, 512)

        # Decoder
        self.decode4 = ConvBNReLU(512, 256)
        self.decode3 = ConvBNReLU(256, 128)
        self.decode2 = ConvBNReLU(128, 64)
        self.decode1 = Conv(64, n_classes)

        # Upsampling
        self.up = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x, x_encode1_indices = self.encode1(x)
        x, x_encode2_indices = self.encode2(x)
        x, x_encode3_indices = self.encode3(x)
        x, x_encode4_indices = self.encode4(x)

        # Decoder
        x = self.decode4(self.up(x, x_encode4_indices))
        x = self.decode3(self.up(x, x_encode3_indices))
        x = self.decode2(self.up(x, x_encode2_indices))
        x = self.decode1(self.up(x, x_encode1_indices))

        return x
