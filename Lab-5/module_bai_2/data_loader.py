import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab, pad_token='<PAD>', unk_token='<UNK>', cls_token='<CLS>', sep_token='<SEP>', config=None):
        if config is not None:
            pad_token = config.TOKENIZER['pad_token']
            unk_token = config.TOKENIZER['unk_token']
            cls_token = config.TOKENIZER['cls_token']
            sep_token = config.TOKENIZER['sep_token']
        self.vocab = vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        self.pad_idx = self.word_to_idx[pad_token]
        self.unk_idx = self.word_to_idx[unk_token]
        self.cls_idx = self.word_to_idx[cls_token]
        self.sep_idx = self.word_to_idx[sep_token]
    
    def encode(self, words, max_length=128, padding='max_length', truncation=True):
        if truncation and len(words) > max_length - 2:
            words = words[:max_length - 2]
        
        input_ids = [self.cls_idx] + [self.word_to_idx.get(word, self.unk_idx) for word in words] + [self.sep_idx]
        
        attention_mask = [1] * len(input_ids)
        
        if padding == 'max_length':
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_idx] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.vocab)

def build_vocab(data_path, min_freq=2, config=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    word_counter = Counter()
    
    for item in data:
        words = item['words']
        word_counter.update(words)
    
    if config is not None:
        vocab = [
            config.TOKENIZER['pad_token'],
            config.TOKENIZER['unk_token'],
            config.TOKENIZER['cls_token'],
            config.TOKENIZER['sep_token']
        ]
    else:
        vocab = ['<PAD>', '<UNK>', '<CLS>', '<SEP>']
    
    vocab.extend([word for word, count in word_counter.items() if count >= min_freq])
    
    return vocab

def build_label_vocab(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    all_labels = []
    for item in data:
        tags = item['tags']
        all_labels.extend(tags)
    
    unique_labels = sorted(list(set(all_labels)))
    unique_labels = ['<PAD>'] + unique_labels
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    return label_to_idx, idx_to_label, unique_labels

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, label_to_idx=None):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        self.words_list = []
        self.tags_list = []
        
        for item in data:
            self.words_list.append(item['words'])
            self.tags_list.append(item['tags'])
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_idx = label_to_idx
        self.pad_label_idx = label_to_idx['<PAD>'] if label_to_idx else 0
    
    def __len__(self):
        return len(self.words_list)
    
    def __getitem__(self, idx):
        words = self.words_list[idx]
        tags = self.tags_list[idx]
        
        if len(words) > self.max_length - 2:
            words = words[:self.max_length - 2]
            tags = tags[:self.max_length - 2]
        
        encoding = self.tokenizer.encode(words, max_length=self.max_length)
        
        label_ids = [self.label_to_idx['O']]
        for tag in tags:
            label_ids.append(self.label_to_idx.get(tag, self.label_to_idx['O']))
        label_ids.append(self.label_to_idx['O'])
        
        padding_length = self.max_length - len(label_ids)
        label_ids = label_ids + [self.pad_label_idx] * padding_length
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def get_data_loaders(train_path, dev_path, test_path, batch_size=32, max_length=128, min_freq=2, config=None):
    vocab = build_vocab(train_path, min_freq, config)
    tokenizer = SimpleTokenizer(vocab, config=config)
    vocab_size = len(tokenizer)
    
    label_to_idx, idx_to_label, unique_labels = build_label_vocab(train_path)
    num_labels = len(unique_labels)
    
    train_dataset = NERDataset(train_path, tokenizer, max_length, label_to_idx)
    dev_dataset = NERDataset(dev_path, tokenizer, max_length, label_to_idx)
    test_dataset = NERDataset(test_path, tokenizer, max_length, label_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader, test_loader, num_labels, label_to_idx, idx_to_label, vocab_size

