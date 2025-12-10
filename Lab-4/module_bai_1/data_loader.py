import json
import torch
from torch.utils.data import Dataset, DataLoader
import random

class TranslationDataset(Dataset):
    def __init__(self, data_path, src_vocab, tgt_vocab, max_length, sample_size=None):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item['english'].lower().strip()
        tgt_text = item['vietnamese'].lower().strip()
        
        src_indices = self.src_vocab.encode(src_text, self.max_length)
        tgt_indices = self.tgt_vocab.encode(tgt_text, self.max_length)
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def create_dataloaders(config, src_vocab, tgt_vocab):
    train_dataset = TranslationDataset(
        config.train_path, src_vocab, tgt_vocab, config.max_length
    )
    dev_dataset = TranslationDataset(
        config.dev_path, src_vocab, tgt_vocab, config.max_length
    )
    test_dataset = TranslationDataset(
        config.test_path, src_vocab, tgt_vocab, config.max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, dev_loader, test_loader

def build_vocabularies(config):
    print("[INFO] Building vocab...")
    
    with open(config.train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    src_sentences = [item['english'].lower().strip() for item in train_data]
    tgt_sentences = [item['vietnamese'].lower().strip() for item in train_data]
    
    src_vocab.build_from_sentences(src_sentences)
    tgt_vocab.build_from_sentences(tgt_sentences)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    return src_vocab, tgt_vocab

from .vocabulary import Vocabulary

