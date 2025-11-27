import torch
from tqdm import tqdm

from utils import cal_metrics

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels, lengths in tqdm(data_loader, desc='Evaluating'):
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            logits = model(inputs, lengths)
            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = cal_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds

