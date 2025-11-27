import torch
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

def cal_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    }
    return metrics

def print_metrics(metrics, prefix=""):
    print(f"{prefix} Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

def plot_training_history(history):
    fig, axes = plt.subplots(1,2,figsize=(15,5))

    axes[0].plot(history["train_loss"], label="Training Loss", marker="o")
    axes[0].plot(history["dev_loss"], label = "Validation Loss", marker = "*")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss History trên train và dev")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_f1"], label="Training F1", marker="o")
    axes[1].plot(history["dev_f1"], label="Validation F1", marker="*")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score History trên train và dev")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)