import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif self.mode == 'max':
            if val_score < self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_score
                self.counter = 0
        else:
            if val_score > self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_score
                self.counter = 0

class Trainer:
    def __init__(self, model, train_loader, dev_loader, device, idx_to_label, learning_rate=2e-5, 
                 num_epochs=10, patience=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.num_epochs = num_epochs
        self.idx_to_label = idx_to_label
        self.pad_label_idx = 0
        for idx, label in idx_to_label.items():
            if label == '<PAD>':
                self.pad_label_idx = idx
                break
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_label_idx)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
    
    def align_predictions(self, predictions, label_ids, attention_mask):
        preds_list = []
        labels_list = []
        
        for pred, label, mask in zip(predictions, label_ids, attention_mask):
            pred_seq = []
            label_seq = []
            for i, (p, l, m) in enumerate(zip(pred, label, mask)):
                if m == 1 and l != self.pad_label_idx:
                    pred_label = self.idx_to_label[p]
                    true_label = self.idx_to_label[l]
                    if pred_label != '<PAD>' and true_label != '<PAD>':
                        if i > 0 and i < len(pred) - 1:
                            pred_seq.append(pred_label)
                            label_seq.append(true_label)
            preds_list.append(pred_seq)
            labels_list.append(label_seq)
        
        return preds_list, labels_list
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                mask_np = attention_mask.cpu().numpy()
                
                preds_list, labels_list = self.align_predictions(preds, labels_np, mask_np)
                all_preds.extend(preds_list)
                all_labels.extend(labels_list)
        
        avg_loss = total_loss / len(self.dev_loader)
        
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        return avg_loss, precision, recall, f1
    
    def train(self):
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch + 1}/{self.num_epochs}')
            
            train_loss = self.train_epoch()
            val_loss, val_precision, val_recall, val_f1 = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['val_f1'].append(val_f1)
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
            
            self.early_stopping(val_f1)
            if self.early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        return self.history
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                mask_np = attention_mask.cpu().numpy()
                
                preds_list, labels_list = self.align_predictions(preds, labels_np, mask_np)
                all_preds.extend(preds_list)
                all_labels.extend(labels_list)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        report = classification_report(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }

