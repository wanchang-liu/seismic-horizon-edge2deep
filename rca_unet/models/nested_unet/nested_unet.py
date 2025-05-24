import torch
from torch import nn
from models.unet.unet_parts import DoubleConv, Down, OutConv


# UNet++
class NestedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False):
        super().__init__()

        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Deep supervision
        self.deep_supervision = deep_supervision

        # First column
        self.down0_0 = DoubleConv(n_channels, 64)
        self.down1_0 = Down(64, 128)
        self.down2_0 = Down(128, 256)
        self.down3_0 = Down(256, 512)
        self.down4_0 = Down(512, 1024)

        # Second column
        self.conv0_1 = DoubleConv(64 + 128, 64)
        self.conv1_1 = DoubleConv(128 + 256, 128)
        self.conv2_1 = DoubleConv(256 + 512, 256)
        self.conv3_1 = DoubleConv(512 + 1024, 512)

        # Third column
        self.conv0_2 = DoubleConv(64 * 2 + 128, 64)
        self.conv1_2 = DoubleConv(128 * 2 + 256, 128)
        self.conv2_2 = DoubleConv(256 * 2 + 512, 256)

        # Fourth column
        self.conv0_3 = DoubleConv(64 * 3 + 128, 64)
        self.conv1_3 = DoubleConv(128 * 3 + 256, 128)

        # Fifth column
        self.conv0_4 = DoubleConv(64 * 4 + 128, 64)

        # Output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, input):
        x0_0 = self.down0_0(input)
        x1_0 = self.down1_0(x0_0)
        x2_0 = self.down2_0(x1_0)
        x3_0 = self.down3_0(x2_0)
        x4_0 = self.down4_0(x3_0)

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.outc(x0_1)
            output2 = self.outc(x0_2)
            output3 = self.outc(x0_3)
            output4 = self.outc(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.outc(x0_4)
            return output
