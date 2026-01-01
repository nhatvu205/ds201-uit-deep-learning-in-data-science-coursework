import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, history['val_precision'], 'g-', label='Precision')
    axes[1, 0].plot(epochs, history['val_recall'], 'orange', label='Recall')
    axes[1, 0].plot(epochs, history['val_f1'], 'purple', label='F1')
    axes[1, 0].set_title('Validation Metrics Through Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, history['train_loss'], 'b-', alpha=0.5, label='Train Loss')
    axes[1, 1].plot(epochs, history['val_loss'], 'r-', alpha=0.5, label='Val Loss')
    axes[1, 1].plot(epochs, history['train_acc'], 'b--', alpha=0.5, label='Train Acc')
    axes[1, 1].plot(epochs, history['val_acc'], 'r--', alpha=0.5, label='Val Acc')
    axes[1, 1].set_title('Combined Training History')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_metrics_through_epochs(history):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['val_acc']) + 1)
    
    ax.plot(epochs, history['val_acc'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(epochs, history['val_precision'], 's-', label='Precision', linewidth=2)
    ax.plot(epochs, history['val_recall'], '^-', label='Recall', linewidth=2)
    ax.plot(epochs, history['val_f1'], 'd-', label='F1-Score', linewidth=2)
    
    ax.set_title('Validation Metrics Through Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def create_metrics_table(history, test_metrics=None):
    data = {
        'Epoch': range(1, len(history['train_loss']) + 1),
        'Train Loss': [f'{x:.4f}' for x in history['train_loss']],
        'Train Acc': [f'{x:.4f}' for x in history['train_acc']],
        'Val Loss': [f'{x:.4f}' for x in history['val_loss']],
        'Val Acc': [f'{x:.4f}' for x in history['val_acc']],
        'Val Precision': [f'{x:.4f}' for x in history['val_precision']],
        'Val Recall': [f'{x:.4f}' for x in history['val_recall']],
        'Val F1': [f'{x:.4f}' for x in history['val_f1']]
    }
    
    df = pd.DataFrame(data)
    
    if test_metrics:
        summary_data = {
            'Metric': ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score'],
            'Value': [
                f"{test_metrics['accuracy']:.4f}",
                f"{test_metrics['precision']:.4f}",
                f"{test_metrics['recall']:.4f}",
                f"{test_metrics['f1']:.4f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        return df, summary_df
    
    return df, None

def display_metrics_table(df, summary_df=None):
    print("\n" + "-"*80)
    print("TRAINING METRICS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    if summary_df is not None:
        print("\n" + "="*80)
        print("TEST SET SUMMARY METRICS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80 + "\n")

