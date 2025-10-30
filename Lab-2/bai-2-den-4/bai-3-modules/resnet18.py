import torch
import torch.nn as nn


class BasicBlock(nn.Module):
   
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Conv1: 7x7 conv, stride 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # MaxPool sau conv1: 3x3, stride 2
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Conv2_x: 2 blocks, 64 channels
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        
        # MaxPool giữa conv2_x và conv3_x
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        # Conv3_x: 2 blocks, 128 channels
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=1)
        
        # MaxPool giữa conv3_x và conv4_x
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        # Conv4_x: 2 blocks, 256 channels
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=1)
        
        # MaxPool giữa conv4_x và conv5_x
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        # Conv5_x: 2 blocks, 512 channels
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=1)
        
        # Global Average Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Tạo một layer gồm các residual blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Conv1 + MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        # Conv2_x
        x = self.layer1(x)
        x = self.maxpool2(x)
        
        # Conv3_x
        x = self.layer2(x)
        x = self.maxpool3(x)
        
        # Conv4_x
        x = self.layer3(x)
        x = self.maxpool4(x)
        
        # Conv5_x
        x = self.layer4(x)
        
        # Global AvgPool + FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def build_resnet18(num_classes):
    return ResNet18(num_classes=num_classes)

