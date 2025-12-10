import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_epoch(model, dataloader, optimizer, criterion, config):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        src = batch['src'].to(config.device)
        tgt = batch['tgt'].to(config.device)
        
        optimizer.zero_grad()
        
        output = model(src, tgt, config.teacher_forcing_ratio)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, config):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            src = batch['src'].to(config.device)
            tgt = batch['tgt'].to(config.device)
            
            output = model(src, tgt, 0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def train_model(model, train_loader, dev_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_valid_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    print("[INFO] Starting training...")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
        valid_loss = evaluate_epoch(model, dev_loader, criterion, config)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(valid_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {valid_loss:.4f}')
        print(f'Learning Rate: {new_lr:.6f}', end='')
        if new_lr < old_lr:
            print(f' (reduced from {old_lr:.6f})')
        else:
            print()
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'Best model updated (val loss: {valid_loss:.4f})')
        else:
            patience_counter += 1
            print(f'[INFO] No improvement. Patience: {patience_counter}/{config.early_stopping_patience}')
            
            if patience_counter >= config.early_stopping_patience:
                print(f'\n[INFO] Early stopping triggered after {epoch+1} epochs')
                print(f'[INFO] Best validation loss: {best_valid_loss:.4f}')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nLoaded best model with validation loss: {best_valid_loss:.4f}')
    
    return train_losses, val_losses

