from torch import nn
from models.segnet.segnet_parts import Encoder_DoubleConv, Encoder_CubicConv, Decoder_CubicConv, Decoder_DoubleConv, \
    Decoder_Classification


class SegNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        # Initialization
        super().__init__()

        # Encoder
        self.encode1 = Encoder_DoubleConv(n_channels, 64)
        self.encode2 = Encoder_DoubleConv(64, 128)
        self.encode3 = Encoder_CubicConv(128, 256)
        self.encode4 = Encoder_CubicConv(256, 512)
        self.encode5 = Encoder_CubicConv(512, 512)

        # Decoder
        self.decode5 = Decoder_CubicConv(512, 512)
        self.decode4 = Decoder_CubicConv(512, 256)
        self.decode3 = Decoder_CubicConv(256, 128)
        self.decode2 = Decoder_DoubleConv(128, 64)
        self.decode1 = Decoder_Classification(64, n_classes)

        # Upsampling
        self.up = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x, x_encode1_indices = self.encode1(x)
        x, x_encode2_indices = self.encode2(x)
        x, x_encode3_indices = self.encode3(x)
        x, x_encode4_indices = self.encode4(x)
        x, x_encode5_indices = self.encode5(x)

        # Decoder
        x = self.decode5(self.up(x, x_encode5_indices))
        x = self.decode4(self.up(x, x_encode4_indices))
        x = self.decode3(self.up(x, x_encode3_indices))
        x = self.decode2(self.up(x, x_encode2_indices))
        x = self.decode1(self.up(x, x_encode1_indices))

        return x
