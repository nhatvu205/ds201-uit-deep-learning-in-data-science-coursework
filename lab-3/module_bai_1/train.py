import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os

from config import Config
from vocab import Vocabulary
from data_loader import get_data_loaders
from model import LSTMClassifier
from utils import cal_metrics, print_metrics, plot_training_history, count_parameters

def train_epoch(model, train_loader, crit, optimizer, device, config):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels, lengths in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        
        logits = model(inputs, lengths)
        loss = crit(logits, labels)
        loss.backward()

        # Tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            "Loss": total_loss
        })

    avg_loss = total_loss/len(train_loader)
    metrics = cal_metrics(all_labels, all_preds)

    return avg_loss, metrics

def train_model(model, train_loader, dev_loader, config):
    device = config.DEVICE
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {
        "train_loss": [],
        "dev_loss": [],
        "train_f1": [],
        "dev_f1": []
    }

    best_f1 = 0.0
    patience_counter = 0

    print("---BẮT ĐẦU TRAINING---")
    print(f"Số epoch: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")

    for epoch in range(config.NUM_EPOCHS):
        print("-"*20)
        print(f"Epoch {epoch +1}/{config.NUM_EPOCHS}")

        train_loss, train_metrics = train_epoch(
            model, train_loader, crit, optimizer, device, config
        )

        from evaluate import evaluate_model
        dev_loss, dev_metrics, _, _ = evaluate_model(
            model, dev_loader, crit, device
        )

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["train_f1"].append(train_metrics["f1_macro"])
        history["dev_f1"].append(dev_metrics["f1_macro"])

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1_macro']:.4f}")
        print(f"Dev Loss: {dev_loss:.4f} | Dev F1: {dev_metrics['f1_macro']:.4f}")

        if dev_metrics['f1_macro'] > best_f1:
            best_f1 = dev_metrics['f1_macro']
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping sau {epoch + 1} epochs")
            break
    print("---HOÀN THÀNH TRAINING---")
    print(f"Best F1: {best_f1:.4f}")

    return model, history