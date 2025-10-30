from typing import Dict, Tuple
import sys
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

        # Update progress bar
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
) -> Dict[str, float]:
    print("=" * 60)
    print("Thiết bị:", device)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    print("=" * 60)
    return {
        "test_accuracy": te_acc,
        "test_precision": te_p,
        "test_recall": te_r,
        "test_f1": te_f1,
    }

