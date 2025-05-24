import torch
from torch import nn
from models.blocks.cbam_block import CBAM
from models.blocks.attention_gate import Attention_Gate
from models.unet.unet_parts import DoubleConv, Down, OutConv, UpsampleMethod, Up_Sampling


class CBAMAttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample_method=UpsampleMethod.Bilinear, ratio=16, kernel_size=7):
        super().__init__()

        self.Inc = DoubleConv(n_channels, 64)
        self.Down1 = Down(64, 128)
        self.Down2 = Down(128, 256)
        self.Down3 = Down(256, 512)
        self.Down4 = Down(512, 1024)

        self.Cbam1 = CBAM(64, ratio, kernel_size)
        self.Cbam2 = CBAM(128, ratio, kernel_size)
        self.Cbam3 = CBAM(256, ratio, kernel_size)
        self.Cbam4 = CBAM(512, ratio, kernel_size)
        self.Cbam5 = CBAM(1024, ratio, kernel_size)

        self.Up4 = Up_Sampling(1024, 512, upsample_method)
        self.Up3 = Up_Sampling(512, 256, upsample_method)
        self.Up2 = Up_Sampling(256, 128, upsample_method)
        self.Up1 = Up_Sampling(128, 64, upsample_method)

        if upsample_method == UpsampleMethod.Bilinear:
            self.Att4 = Attention_Gate(1024, 512, 256)
            self.Att3 = Attention_Gate(512, 256, 128)
            self.Att2 = Attention_Gate(256, 128, 64)
            self.Att1 = Attention_Gate(128, 64, 32)

            self.Up_Conv4 = DoubleConv(512 + 1024, 512)
            self.Up_Conv3 = DoubleConv(256 + 512, 256)
            self.Up_Conv2 = DoubleConv(128 + 256, 128)
            self.Up_Conv1 = DoubleConv(64 + 128, 64)

        else:
            self.Att4 = Attention_Gate(512, 512, 256)
            self.Att3 = Attention_Gate(256, 256, 128)
            self.Att2 = Attention_Gate(128, 128, 64)
            self.Att1 = Attention_Gate(64, 64, 32)

            self.Up_Conv4 = DoubleConv(1024, 512)
            self.Up_Conv3 = DoubleConv(512, 256)
            self.Up_Conv2 = DoubleConv(256, 128)
            self.Up_Conv1 = DoubleConv(128, 64)

        self.Outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder+CBAM
        x1 = self.Inc(x)
        cbam1 = self.Cbam1(x1)

        x2 = self.Down1(x1)
        cbam2 = self.Cbam2(x2)

        x3 = self.Down2(x2)
        cbam3 = self.Cbam3(x3)

        x4 = self.Down3(x3)
        cbam4 = self.Cbam4(x4)

        x5 = self.Down4(x4)
        cbam5 = self.Cbam5(x5)

        # Decoder
        x = self.Up4(cbam5)
        x4 = self.Att4(x, cbam4)
        x = torch.cat((x4, x), dim=1)
        x = self.Up_Conv4(x)

        x = self.Up3(x)
        x3 = self.Att3(x, cbam3)
        x = torch.cat((x3, x), dim=1)
        x = self.Up_Conv3(x)

        x = self.Up2(x)
        x2 = self.Att2(x, cbam2)
        x = torch.cat((x2, x), dim=1)
        x = self.Up_Conv2(x)

        x = self.Up1(x)
        x1 = self.Att1(x, cbam1)
        x = torch.cat((x1, x), dim=1)
        x = self.Up_Conv1(x)

        return self.Outc(x)
