import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_training_history(history: Dict[str, List[float]]):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    axes[0, 0].plot(epochs, history['test_loss'], 'r-o', label='Test')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train')
    axes[0, 1].plot(epochs, history['test_acc'], 'r-o', label='Test')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(epochs, history['train_precision'], 'b-o', label='Train')
    axes[1, 0].plot(epochs, history['test_precision'], 'r-o', label='Test')
    axes[1, 0].set_title('Precision (macro)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1
    axes[1, 1].plot(epochs, history['train_f1'], 'b-o', label='Train')
    axes[1, 1].plot(epochs, history['test_f1'], 'r-o', label='Test')
    axes[1, 1].set_title('F1 Score (macro)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def show_predictions(model, test_loader, class_names, device, num_correct=2, num_wrong=2):
    model.eval()
    
    correct_samples = []
    wrong_samples = []
    
    # Denormalize để hiển thị ảnh
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            preds = torch.argmax(outputs, dim=1).cpu()
            
            for i in range(len(images)):
                img = images[i]
                true_label = labels[i].item()
                pred_label = preds[i].item()
                
                # Denormalize
                img_denorm = img * std + mean
                img_denorm = torch.clamp(img_denorm, 0, 1)
                img_np = img_denorm.permute(1, 2, 0).numpy()
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append((img_np, true_label, pred_label))
                elif true_label != pred_label and len(wrong_samples) < num_wrong:
                    wrong_samples.append((img_np, true_label, pred_label))
                
                if len(correct_samples) >= num_correct and len(wrong_samples) >= num_wrong:
                    break
            
            if len(correct_samples) >= num_correct and len(wrong_samples) >= num_wrong:
                break
    
    # Vẽ
    total_samples = len(correct_samples) + len(wrong_samples)
    fig, axes = plt.subplots(1, total_samples, figsize=(4 * total_samples, 4))
    
    if total_samples == 1:
        axes = [axes]
    
    idx = 0
    
    # Dự đoán đúng
    for img, true_label, pred_label in correct_samples:
        axes[idx].imshow(img)
        axes[idx].set_title(f'✓ Đúng\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}', 
                           color='green', fontsize=10)
        axes[idx].axis('off')
        idx += 1
    
    # Dự đoán sai
    for img, true_label, pred_label in wrong_samples:
        axes[idx].imshow(img)
        axes[idx].set_title(f'✗ Sai\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}', 
                           color='red', fontsize=10)
        axes[idx].axis('off')
        idx += 1
    
    plt.tight_layout()
    plt.show()

