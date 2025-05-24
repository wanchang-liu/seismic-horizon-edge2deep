from torch import nn
from models.blocks.cbam_block import CBAM
from models.unet.unet_parts import DoubleConv, Down, OutConv, UpsampleMethod, Up_Bilinear, Up_Transposed, \
    Up_BilinearConv


class CBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample_method=UpsampleMethod.Bilinear, ratio=16, kernel_size=7):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.cbam1 = CBAM(64, ratio, kernel_size)

        self.down1 = Down(64, 128)
        self.cbam2 = CBAM(128, ratio, kernel_size)

        self.down2 = Down(128, 256)
        self.cbam3 = CBAM(256, ratio, kernel_size)

        self.down3 = Down(256, 512)
        self.cbam4 = CBAM(512, ratio, kernel_size)

        self.down4 = Down(512, 1024)
        self.cbam5 = CBAM(1024, ratio, kernel_size)

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

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder+CBAM
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)

        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)

        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)

        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)

        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)

        # Decoder
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)

        return self.outc(x)
