import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
  """
  Kiến trúc mô hình LeNet:
  - Input: ảnh grayscale kích thước 28x28 pixels
  - Conv1: 5x5 kernel, 6 filters, padding 2
  - AvgPool1: 2x2, stride 2
  - Conv2: 5x5 kernel, 16 filters
  - AvgPool2: 2x2, stride 2
  - FC1: 120 units
  - FC2: 84 units
  - FC3: 10 units (output)
  """

  def __init__(self):
    super(LeNet, self).__init__()

    # Lớp convolutional
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2) # Ảnh grayscale nên input có 1 channel màu
    self.conv2 = nn.Conv2d(in_channels= 6, out_channels=16, kernel_size=5, padding=0) # padding=0 để thu nhỏ ảnh 

    # Lớp pooling
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    # Lớp Fully Connected
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84,10)

  def forward(self, x):
    # Conv1 -> ReLU -> AvgPool
    x = self.pool(F.relu(self.conv1(x)))

    # Conv2 -> ReLU -> AvgPool
    x = self.pool(F.relu(self.conv2(x)))

    # Flatten
    x = x.view(-1, 16*5*5)

    # FC1 -> Sigmoid
    x = F.sigmoid(self.fc1(x))

    # FC2 -> Sigmoid
    x = F.sigmoid(self.fc2(x))

    # FC3 (output)
    x = self.fc3(x)

    return x

  def get_model_summary(self):
    """Tóm tắt kiến trúc mô hình"""
    print("LeNet model summary")
    print("-"*50)
    total_params = sum(p.numel() for p in self.parameters()) # tổng tham số
    print(f"Tổng tham số: {total_params:,}")
