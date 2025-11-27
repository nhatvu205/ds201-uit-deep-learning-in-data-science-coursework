import torch
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_ner_metrics(y_true_flat, y_pred_flat, tag_names, ignore_tag='O'):
    mask = [tag != ignore_tag for tag in y_true_flat]
    y_true_filtered = [tag for tag, m in zip(y_true_flat, mask) if m]
    y_pred_filtered = [tag for tag, m in zip(y_pred_flat, mask) if m]
    
    if len(y_true_filtered) == 0:
        return {'f1_macro': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    f1_macro = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    precision = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    
    return {
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    }

def print_metrics(metrics, prefix=''):
    print(f'{prefix} F1: {metrics["f1_macro"]:.4f} | Precision: {metrics["precision"]:.4f} | Recall: {metrics["recall"]:.4f}')

def plot_training_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['dev_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss History')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_f1'], label='Train F1', marker='o')
    axes[1].plot(history['dev_f1'], label='Val F1', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score History')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'[INFO] Saved plot to {save_path}')
    plt.close()

def save_history(history, save_path='training_history.json'):
    import json
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[INFO] Saved history to {save_path}')

