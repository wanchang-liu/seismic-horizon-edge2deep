from torch import nn
from models.unet.unet_parts import DoubleConv, OutConv, UpsampleMethod, Up_Bilinear, Up_Transposed, Up_BilinearConv


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample_method=UpsampleMethod.Bilinear):
        super().__init__()

        self.Conv1 = DoubleConv(n_channels, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)
        self.Conv5 = DoubleConv(512, 1024)

        self.W1 = nn.Conv2d(n_channels, 64, kernel_size=1)
        self.W2 = nn.Conv2d(64, 128, kernel_size=1)
        self.W3 = nn.Conv2d(128, 256, kernel_size=1)
        self.W4 = nn.Conv2d(256, 512, kernel_size=1)
        self.W5 = nn.Conv2d(512, 1024, kernel_size=1)

        self.Pool1 = nn.MaxPool2d(2)
        self.Pool2 = nn.MaxPool2d(2)
        self.Pool3 = nn.MaxPool2d(2)
        self.Pool4 = nn.MaxPool2d(2)

        if upsample_method == UpsampleMethod.Bilinear:
            self.Up4 = Up_Bilinear(512 + 1024, 512)
            self.Up3 = Up_Bilinear(256 + 512, 256)
            self.Up2 = Up_Bilinear(128 + 256, 128)
            self.Up1 = Up_Bilinear(64 + 128, 64)
        elif upsample_method == UpsampleMethod.Transposed:
            self.Up4 = Up_Transposed(1024, 512)
            self.Up3 = Up_Transposed(512, 256)
            self.Up2 = Up_Transposed(256, 128)
            self.Up1 = Up_Transposed(128, 64)
        elif upsample_method == UpsampleMethod.BilinearConv:
            self.Up4 = Up_BilinearConv(1024, 512)
            self.Up3 = Up_BilinearConv(512, 256)
            self.Up2 = Up_BilinearConv(256, 128)
            self.Up1 = Up_BilinearConv(128, 64)

        self.Outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder+Residual
        x1 = self.Conv1(x) + self.W1(x)

        x2 = self.Pool1(x1)
        x2 = self.Conv2(x2) + self.W2(x2)

        x3 = self.Pool2(x2)
        x3 = self.Conv3(x3) + self.W3(x3)

        x4 = self.Pool3(x3)
        x4 = self.Conv4(x4) + self.W4(x4)

        x5 = self.Pool4(x4)
        x5 = self.Conv5(x5) + self.W5(x5)

        # 解码层
        x = self.Up4(x5, x4)
        x = self.Up3(x, x3)
        x = self.Up2(x, x2)
        x = self.Up1(x, x1)

        return self.Outc(x)
