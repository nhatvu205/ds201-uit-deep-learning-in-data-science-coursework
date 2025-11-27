import torch
from tqdm import tqdm

from utils import cal_ner_metrics

def evaluate_model(model, data_loader, device, config):
    model.eval()
    all_preds, all_labels = [], []
    
    pbar = tqdm(data_loader, desc='Evaluating', ncols=100)
    
    with torch.no_grad():
        for inputs, tags, lengths in pbar:
            inputs, tags, lengths = inputs.to(device), tags.to(device), lengths.to(device)
            logits = model(inputs, lengths)
            predictions = torch.argmax(logits, dim=2)
            
            for i in range(len(lengths)):
                length = lengths[i].item()
                pred_tags = predictions[i][:length].cpu().tolist()
                true_tags = tags[i][:length].cpu().tolist()
                all_preds.extend([config.ID2TAG[p] for p in pred_tags])
                all_labels.extend([config.ID2TAG[t] for t in true_tags])
    
    metrics = cal_ner_metrics(all_labels, all_preds, config.TAGS)
    return metrics, all_labels, all_preds

