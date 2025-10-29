import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from mnist_dataset import MnistDataset, collate_fn
from perceptron_3_layer import MLP3Layer, train_epoch, evaluate
from google.colab import drive
from sklearn.metrics import confusion_matrix

def plot_training_history(train_history, test_history):
    """Vẽ đồ thị quá trình training"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if idx < len(axes):
            axes[idx].plot(train_history[metric], label='Train', marker='o')
            axes[idx].plot(test_history[metric], label='Test', marker='s')
            axes[idx].set_title(f'{metric.capitalize()} over epochs')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    # Ẩn subplot cuối (vì chỉ có 5 metrics)
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()

def print_final_results(test_metrics):
    """In kết quả chi tiết cho từng chữ số"""
    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH MLP 3-LAYER VỚI ReLU VÀ SOFTMAX")
    print("="*60)
    
    print("\nKết quả tổng quát:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision (macro): {test_metrics['precision']:.4f}")
    print(f"Recall (macro): {test_metrics['recall']:.4f}")
    print(f"F1-score (macro): {test_metrics['f1']:.4f}")
    
    print("\nKết quả chi tiết cho từng chữ số:")
    print("-"*60)
    print("Chữ số | Precision |  Recall  | F1-Score")
    print("-"*60)
    
    for digit in range(10):
        print(f"   {digit}   |   {test_metrics['per_digit_metrics']['precision'][digit]:.4f}  |  "
              f"{test_metrics['per_digit_metrics']['recall'][digit]:.4f}  |  "
              f"{test_metrics['per_digit_metrics']['f1'][digit]:.4f}")
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - MLP 3-Layer')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
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
    print("Loading datasets...")
    train_dataset = MnistDataset(train_images_path, train_labels_path)
    test_dataset = MnistDataset(test_images_path, test_labels_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Có thể tăng batch size cho MLP lớn hơn
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Thiết lập hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 28 * 28
    hidden_size1 = 128  # Số neurons trong hidden layer 1
    hidden_size2 = 64   # Số neurons trong hidden layer 2
    num_classes = 10
    learning_rate = 0.01
    num_epochs = 15

    print("\n" + "="*60)
    print("BÀI 2: HUẤN LUYỆN MÔ HÌNH MLP 3-LAYER")
    print("="*60)
    print("\nThông số mô hình:")
    print(f"- Kiến trúc: {input_size} -> {hidden_size1} (ReLU) -> {hidden_size2} (ReLU) -> {num_classes} (Softmax)")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Batch size: {train_loader.batch_size}")
    print(f"- Số epoch: {num_epochs}")
    print(f"- Device: {device}")
    print(f"- Optimizer: SGD")
    print("="*60)

    # Khởi tạo model
    model = MLP3Layer(input_size, hidden_size1, hidden_size2, num_classes).to(device)
    
    # In thông tin model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTổng số tham số: {total_params:,}")
    print(f"Số tham số trainable: {trainable_params:,}")

    # Loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Lưu lịch sử training
    train_history = {metric: [] for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']}
    test_history = {metric: [] for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']}

    # Training loop
    print("\nBắt đầu huấn luyện...")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*60)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training   - Loss: {train_metrics['loss']:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")

        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Validation - Loss: {test_metrics['loss']:.4f}, "
              f"Accuracy: {test_metrics['accuracy']:.4f}, "
              f"F1: {test_metrics['f1']:.4f}")

        # Lưu metrics
        for metric in train_history:
            train_history[metric].append(train_metrics[metric])
            test_history[metric].append(test_metrics[metric])

    # Hiển thị kết quả cuối cùng
    print("\n" + "="*60)
    print("HUẤN LUYỆN HOÀN TẤT!")
    print("="*60)
    print_final_results(test_metrics)
    
    # Vẽ đồ thị training history
    print("\nVẽ đồ thị quá trình training...")
    plot_training_history(train_history, test_history)
    
    # Vẽ confusion matrix
    print("\nVẽ confusion matrix...")
    plot_confusion_matrix(model, test_loader, device)
    
    # So sánh kết quả best epoch
    best_epoch = np.argmax(test_history['accuracy']) + 1
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best test accuracy: {max(test_history['accuracy']):.4f}")
    print(f"Best test F1-score: {test_history['f1'][best_epoch-1]:.4f}")

if __name__ == "__main__":
    main()
