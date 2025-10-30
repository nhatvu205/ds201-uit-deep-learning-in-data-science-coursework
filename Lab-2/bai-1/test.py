import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, device):
    """
    Đánh giá mô hình trên tập test với các độ đo:
    - Precision
    - Recall
    - F1-macro
    
    Args:
        model: mô hình đã được train
        test_loader: DataLoader cho tập test
        device: cpu
    
    Returns:
        metrics: dict chứa các độ đo
        all_predictions: tất cả dự đoán
        all_labels: Tất cả nhãn của ground truth
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    
    print("=" * 60)
    print("\nTEST STARTED")
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Tính các metrics
    precision_macro = precision_score(all_labels, all_predictions, average='macro')
    recall_macro = recall_score(all_labels, all_predictions, average='macro')
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    # Tính accuracy
    accuracy = np.mean(all_predictions == all_labels) * 100
    
    # Tính precision, recall, f1 cho từng lớp
    precision_per_class = precision_score(all_labels, all_predictions, average=None)
    recall_per_class = recall_score(all_labels, all_predictions, average=None)
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }
    
    return metrics, all_predictions, all_labels

def print_evaluation_results(metrics):
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LENET TRÊN MNIST")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    print("=" * 60)
    
    print("\nKẾT QUẢ CHI TIẾT THEO TỪNG LỚP:")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for i in range(10):
        print(f"{i:<10} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} "
              f"{metrics['f1_per_class'][i]:<12.4f}")
    
    print("=" * 60)

def plot_confusion_matrix(all_labels, all_predictions, save_path=None):
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - LeNet on MNIST')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nĐã lưu confusion matrix tại: {save_path}")
    
    plt.show()

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu training history tại: {save_path}")
    
    plt.show()