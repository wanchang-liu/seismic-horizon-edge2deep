import torch
from torch import nn
import torch.nn.functional as F


# 整体嵌套边缘检测模型
class HED_DS(nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.netVggTwo = torch.nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.netVggThr = torch.nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.netVggFou = torch.nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.netVggFiv = torch.nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.netScoreOne = nn.Conv2d(64, 1, kernel_size=1)
        self.netScoreTwo = nn.Conv2d(128, 1, kernel_size=1)
        self.netScoreThr = nn.Conv2d(256, 1, kernel_size=1)
        self.netScoreFou = nn.Conv2d(512, 1, kernel_size=1)
        self.netScoreFiv = nn.Conv2d(512, 1, kernel_size=1)
        self.netCombine = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, tenInput):
        img_H, img_W = tenInput.shape[2], tenInput.shape[3]

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = F.interpolate(tenScoreOne, size=(img_H, img_W), mode='bilinear', align_corners=False)
        tenScoreTwo = F.interpolate(tenScoreTwo, size=(img_H, img_W), mode='bilinear', align_corners=False)
        tenScoreThr = F.interpolate(tenScoreThr, size=(img_H, img_W), mode='bilinear', align_corners=False)
        tenScoreFou = F.interpolate(tenScoreFou, size=(img_H, img_W), mode='bilinear', align_corners=False)
        tenScoreFiv = F.interpolate(tenScoreFiv, size=(img_H, img_W), mode='bilinear', align_corners=False)
        tenScoreCombin = self.netCombine(
            torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))

        # 对每个输出应用sigmoid
        tenScoreOne = torch.sigmoid(tenScoreOne)
        tenScoreTwo = torch.sigmoid(tenScoreTwo)
        tenScoreThr = torch.sigmoid(tenScoreThr)
        tenScoreFou = torch.sigmoid(tenScoreFou)
        tenScoreFiv = torch.sigmoid(tenScoreFiv)
        tenScoreCombin = torch.sigmoid(tenScoreCombin)

        # 直接返回各个输出
        return tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv, tenScoreCombin


# 创建模型
def create_model_ds(args):
    return HED_DS().to(args.device)
