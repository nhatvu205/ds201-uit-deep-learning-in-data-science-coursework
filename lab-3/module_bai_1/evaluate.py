import torch
from tqdm import tqdm
from utils import cal_metrics

def evaluate_model(model, data_loader, crit, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, lengths in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            logits = model(inputs, lengths)
            loss = crit(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    metrics = cal_metrics(all_labels, all_preds)
    
    return avg_loss, metrics, all_labels, all_preds