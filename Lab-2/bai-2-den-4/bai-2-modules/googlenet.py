from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int] | int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, bias=True, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):

    def __init__(
        self,
        in_ch: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ):
        super().__init__()
        # nhánh 1: 1x1
        self.branch1 = BasicConv2d(in_ch, ch1x1, kernel_size=1)

        # nhánh 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        # nhánh 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            BasicConv2d(in_ch, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        # nhánh 4: 3x3 maxpool -> 1x1
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.branch4_proj = BasicConv2d(in_ch, pool_proj, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4_proj(self.branch4_pool(x))
        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout_p: float = 0.4):
        super().__init__()
        # conv7x7 s2, pad=3 (theo yêu cầu)
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception 3a, 3b
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception 4a - 4e
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception 5a, 5b
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # đầu vào 224x224
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_googlenet(num_classes: int) -> nn.Module:
    return GoogLeNet(num_classes=num_classes)


