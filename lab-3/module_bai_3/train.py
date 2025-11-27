import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import cal_ner_metrics

def train_epoch(model, train_loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(train_loader, desc='Training', ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for inputs, tags, lengths in pbar:
        inputs, tags, lengths = inputs.to(device), tags.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs, lengths)
        
        logits_flat = logits.view(-1, config.NUM_TAGS)
        tags_flat = tags.view(-1)
        
        loss = criterion(logits_flat, tags_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=2)
        
        for i in range(len(lengths)):
            length = lengths[i].item()
            pred_tags = predictions[i][:length].cpu().tolist()
            true_tags = tags[i][:length].cpu().tolist()
            all_preds.extend([config.ID2TAG[p] for p in pred_tags])
            all_labels.extend([config.ID2TAG[t] for t in true_tags])
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    metrics = cal_ner_metrics(all_labels, all_preds, config.TAGS)
    return avg_loss, metrics

def evaluate(model, data_loader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(data_loader, desc='Evaluating', ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
    
    with torch.no_grad():
        for inputs, tags, lengths in pbar:
            inputs, tags, lengths = inputs.to(device), tags.to(device), lengths.to(device)
            
            logits = model(inputs, lengths)
            
            logits_flat = logits.view(-1, config.NUM_TAGS)
            tags_flat = tags.view(-1)
            loss = criterion(logits_flat, tags_flat)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=2)
            
            for i in range(len(lengths)):
                length = lengths[i].item()
                pred_tags = predictions[i][:length].cpu().tolist()
                true_tags = tags[i][:length].cpu().tolist()
                all_preds.extend([config.ID2TAG[p] for p in pred_tags])
                all_labels.extend([config.ID2TAG[t] for t in true_tags])
    
    avg_loss = total_loss / len(data_loader)
    metrics = cal_ner_metrics(all_labels, all_preds, config.TAGS)
    return avg_loss, metrics

def train_model(model, train_loader, dev_loader, config):
    device = config.DEVICE
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.TAG2ID['O'])
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                          weight_decay=config.WEIGHT_DECAY)
    
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config.SCHEDULER_FACTOR, 
            patience=config.SCHEDULER_PATIENCE
        )
    
    history = {
        'train_loss': [], 'dev_loss': [],
        'train_f1': [], 'dev_f1': []
    }
    
    best_f1 = 0.0
    patience_counter = 0
    
    print(f'\n{"-"*60}')
    print('TRAINING')
    print(f'{"-"*60}')
    print(f'Epochs: {config.NUM_EPOCHS} | Batch size: {config.BATCH_SIZE}')
    print(f'Learning rate: {config.LEARNING_RATE} | Weight decay: {config.WEIGHT_DECAY}')
    print(f'{"-"*60}\n')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f'{"-"*20}')
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
        dev_loss, dev_metrics = evaluate(model, dev_loader, criterion, device, config)
        
        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_loss)
        history['train_f1'].append(train_metrics['f1_macro'])
        history['dev_f1'].append(dev_metrics['f1_macro'])
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1_macro']:.4f}")
        print(f"Val Loss: {dev_loss:.4f} | Val F1: {dev_metrics['f1_macro']:.4f}")
        
        if scheduler:
            scheduler.step(dev_metrics['f1_macro'])
        
        if dev_metrics['f1_macro'] > best_f1:
            best_f1 = dev_metrics['f1_macro']
            patience_counter = 0
            print(f'New best F1: {best_f1:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement ({patience_counter}/{config.PATIENCE})')
        
        if patience_counter >= config.PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break
        print()
    
    print(f'{"-"*60}')
    print(f'TRAINING COMPLETE | Best F1: {best_f1:.4f}')
    print(f'{"-"*60}')
    
    return model, history

