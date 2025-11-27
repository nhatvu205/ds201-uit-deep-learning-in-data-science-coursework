import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
import numpy as np

from config import Config
from vocab import Vocabulary

class VSFCDataset(Dataset):

    def __init__(self, data_path, vocab, config, max_length=None):
        self.vocab = vocab
        self.config = config
        self.max_length = max_length or config.MAX_SEQ_LENGTH

        self.data = self.load_data(data_path)

    def load_data(self, data_path) -> List[Dict]:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[INFO] Load {data_path} thành công. Kích thước: {len(data)} ")

        label_counts = {}
        for item in data:
            label = item['topic']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Phân bố nhãn: {label_counts}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        sentence = item['sentence']
        label = item['topic']

        input_ids = self.vocab.encode(sentence)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        label_id = self.config.LABEL2ID[label]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)

def collate_fn(batch, pad_idx=0):
    inputs, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    labels = torch.stack(labels)
    return padded_inputs, labels, lengths

def get_data_loaders(vocab, config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = VSFCDataset(config.TRAIN_PATH, vocab, config)
    dev_dataset = VSFCDataset(config.DEV_PATH, vocab, config)
    test_dataset = VSFCDataset(config.TEST_PATH, vocab, config)
    pad_idx = vocab.word2idx[config.PAD_TOKEN]

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, pad_idx = pad_idx))

    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_idx = pad_idx))

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_idx = pad_idx))

    return train_loader, dev_loader, test_loader