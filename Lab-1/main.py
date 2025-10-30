import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from mnist_dataset import MnistDataset, collate_fn
from perceptron_1_layer import MLP, train_epoch, evaluate
from google.colab import drive
from sklearn.metrics import confusion_matrix

def plot_training_history(train_history, test_history):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if idx < len(axes):
            axes[idx].plot(train_history[metric], label='Train')
            axes[idx].plot(test_history[metric], label='Test')
            axes[idx].set_title(f'{metric.capitalize()} over epochs')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].legend()

    plt.tight_layout()
    plt.show()

def print_final_results(test_metrics):
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH MLP 1-LAYER VỚI SOFTMAX")
    print("="*50)
    
    print("\nKết quả tổng quát:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision (macro): {test_metrics['precision']:.4f}")
    print(f"Recall (macro): {test_metrics['recall']:.4f}")
    print(f"F1-score (macro): {test_metrics['f1']:.4f}")
    
    print("\nKết quả chi tiết:")
    print("-"*60)
    print("Chữ số | Precision |  Recall  | F1-Score")
    print("-"*60)
    
    for digit in range(10):
        print(f"   {digit}   |   {test_metrics['per_digit_metrics']['precision'][digit]:.4f}  |  {test_metrics['per_digit_metrics']['recall'][digit]:.4f}  |  {test_metrics['per_digit_metrics']['f1'][digit]:.4f}")
    print("-"*60)

def plot_confusion_matrix(model, test_loader, device):
    """Vẽ confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label']
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():
    # Mount Google Drive
    drive.mount('/content/drive')

    # Đường dẫn dataset
    train_images_path = '/content/drive/MyDrive/Uni/Ki-5-Nam-3/DS201/Lab-1/train-images-idx3-ubyte.gz'
    train_labels_path = '/content/drive/MyDrive/Uni/Ki-5-Nam-3/DS201/Lab-1/train-labels-idx1-ubyte.gz'
    test_images_path = '/content/drive/MyDrive/Uni/Ki-5-Nam-3/DS201/Lab-1/t10k-images-idx3-ubyte.gz'
    test_labels_path = '/content/drive/MyDrive/Uni/Ki-5-Nam-3/DS201/Lab-1/t10k-labels-idx1-ubyte.gz'

    # Tạo dataset và dataloader
    train_dataset = MnistDataset(train_images_path, train_labels_path)
    test_dataset = MnistDataset(test_images_path, test_labels_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Thiết lập model và training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 28 * 28
    num_classes = 10
    learning_rate = 0.01
    num_epochs = 10

    print("\nBắt đầu huấn luyện mô hình MLP 1-layer với:")
    print(f"- Input size: {input_size}")
    print(f"- Output size: {num_classes}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Số epoch: {num_epochs}")
    print(f"- Device: {device}")

    model = MLP(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Lưu lịch sử training
    train_history = {metric: [] for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']}
    test_history = {metric: [] for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']}

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")

        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"\nTest - Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

        # Lưu metrics
        for metric in train_history:
            train_history[metric].append(train_metrics[metric])
            test_history[metric].append(test_metrics[metric])

    # Hiển thị kết quả cuối cùng
    print_final_results(test_metrics)
    
    # Vẽ đồ thị training history
    plot_training_history(train_history, test_history)
    
    # Vẽ confusion matrix
    plot_confusion_matrix(model, test_loader, device)

if __name__ == "__main__":
    main()
