from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics import update_confusion, precision_recall_f1, accuracy
from tqdm import tqdm


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str = "Batch",
) -> Tuple[float, np.ndarray]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    num_classes = len(loader.dataset.classes) if hasattr(loader.dataset, "classes") else None
    if num_classes is None:
        raise ValueError("Sai dataset")
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    pbar = tqdm(loader, desc=desc)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        conf = update_confusion(conf, preds, labels)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    total_loss = total_loss / max(len(loader.dataset), 1)
    return total_loss, conf


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | torch.device = "cuda",
) -> Tuple[Dict[str, float], Dict[str, list]]:
    print("=" * 60)
    print("Device:", device)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'test_loss': [],
        'test_acc': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
    }

    for epoch in range(1, epochs + 1):
        print("-" * 60)
        
        train_loss, train_conf = run_epoch(
            model, train_loader, criterion, optimizer, device, desc=f"Epoch {epoch}/{epochs} [Train]"
        )
        tr_acc = accuracy(train_conf)
        tr_p, tr_r, tr_f1 = precision_recall_f1(train_conf)
        print(f"Train - loss: {train_loss:.4f} | acc: {tr_acc:.4f} | P/R/F1: {tr_p:.4f}/{tr_r:.4f}/{tr_f1:.4f}")

        with torch.no_grad():
            test_loss, test_conf = run_epoch(
                model, test_loader, criterion, None, device, desc=f"Epoch {epoch}/{epochs} [Test]"
            )
        te_acc = accuracy(test_conf)
        te_p, te_r, te_f1 = precision_recall_f1(test_conf)
        print(f"Test  - loss: {test_loss:.4f} | acc: {te_acc:.4f} | P/R/F1: {te_p:.4f}/{te_r:.4f}/{te_f1:.4f}")

        # Lưu vào history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(tr_acc)
        history['train_precision'].append(tr_p)
        history['train_recall'].append(tr_r)
        history['train_f1'].append(tr_f1)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(te_acc)
        history['test_precision'].append(te_p)
        history['test_recall'].append(te_r)
        history['test_f1'].append(te_f1)

    print("=" * 60)
    results = {
        "test_accuracy": te_acc,
        "test_precision": te_p,
        "test_recall": te_r,
        "test_f1": te_f1,
    }
    return results, history

