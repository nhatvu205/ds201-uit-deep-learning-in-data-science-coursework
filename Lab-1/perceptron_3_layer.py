import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.notebook import tqdm

class MLP3Layer(nn.Module):
    def __init__(self, input_size=784, hidden_size1=128, hidden_size2=64, num_classes=10):
        """
        MLP 3 layer với ReLU và Softmax
        Args:
            input_size: Kích thước input (784 cho MNIST)
            hidden_size1: Số neurons trong hidden layer 1
            hidden_size2: Số neurons trong hidden layer 2
            num_classes: Số lượng classes (10 cho MNIST)
        """
        super(MLP3Layer, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # Layer 1: Input -> Hidden 1
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Hidden 1 -> Hidden 2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Hidden 2 -> Output
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass qua 3 layers
        """
        # Flatten input
        x = self.flatten(x)  # (batch, 1, 28, 28) -> (batch, 784)
        
        # Layer 1 với ReLU
        x = self.fc1(x)      # (batch, 784) -> (batch, hidden_size1)
        x = self.relu1(x)    # Áp dụng ReLU activation
        
        # Layer 2 với ReLU
        x = self.fc2(x)      # (batch, hidden_size1) -> (batch, hidden_size2)
        x = self.relu2(x)    # Áp dụng ReLU activation
        
        # Layer 3 với Softmax
        x = self.fc3(x)      # (batch, hidden_size2) -> (batch, num_classes)
        x = self.softmax(x)  # Áp dụng Softmax activation
        
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Huấn luyện model trong 1 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc='Training'):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass và optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Lưu predictions và labels
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Tính metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate(model, test_loader, criterion, device):
    """Đánh giá model trên tập test"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    # Tính metrics cho từng class
    precision_per_digit, recall_per_digit, f1_per_digit, _ = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

    return {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_digit_metrics': {
            'precision': precision_per_digit,
            'recall': recall_per_digit,
            'f1': f1_per_digit
        }
    }
