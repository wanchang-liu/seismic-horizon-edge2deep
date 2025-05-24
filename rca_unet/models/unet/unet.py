from torch import nn
from models.unet.unet_parts import DoubleConv, Down, OutConv, UpsampleMethod, Up_Bilinear, Up_Transposed, \
    Up_BilinearConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample_method):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        if upsample_method == UpsampleMethod.Bilinear:
            self.up1 = Up_Bilinear(512 + 1024, 512)
            self.up2 = Up_Bilinear(256 + 512, 256)
            self.up3 = Up_Bilinear(128 + 256, 128)
            self.up4 = Up_Bilinear(64 + 128, 64)
        elif upsample_method == UpsampleMethod.Transposed:
            self.up1 = Up_Transposed(1024, 512)
            self.up2 = Up_Transposed(512, 256)
            self.up3 = Up_Transposed(256, 128)
            self.up4 = Up_Transposed(128, 64)
        elif upsample_method == UpsampleMethod.BilinearConv:
            self.up1 = Up_BilinearConv(1024, 512)
            self.up2 = Up_BilinearConv(512, 256)
            self.up3 = Up_BilinearConv(256, 128)
            self.up4 = Up_BilinearConv(128, 64)

        # Output
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
