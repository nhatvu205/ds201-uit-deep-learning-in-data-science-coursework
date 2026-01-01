import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import re
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
    
    def encode(self, text, max_length=128, padding='max_length', truncation=True):
        words = self._tokenize(text)
        
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
    
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return words
    
    def __len__(self):
        return len(self.vocab)

def build_vocab(data_path, min_freq=2, config=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    word_counter = Counter()
    
    for item in data.values():
        review = item['review'].lower()
        review = re.sub(r'[^\w\s]', ' ', review)
        words = review.split()
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

class DomainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, label_encoder=None):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.reviews = []
        self.domains = []
        
        for item in data.values():
            self.reviews.append(item['review'])
            self.domains.append(item['domain'])
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.domain_labels = self.label_encoder.fit_transform(self.domains)
            self.num_classes = len(self.label_encoder.classes_)
        else:
            self.label_encoder = label_encoder
            self.domain_labels = self.label_encoder.transform(self.domains)
            self.num_classes = len(self.label_encoder.classes_)
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        domain = self.domain_labels[idx]
        
        encoding = self.tokenizer.encode(review, max_length=self.max_length)
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(domain, dtype=torch.long)
        }

def get_data_loaders(train_path, dev_path, test_path, batch_size=32, max_length=128, min_freq=2, config=None):
    vocab = build_vocab(train_path, min_freq, config)
    tokenizer = SimpleTokenizer(vocab, config=config)
    vocab_size = len(tokenizer)
    
    train_dataset = DomainDataset(train_path, tokenizer, max_length)
    dev_dataset = DomainDataset(dev_path, tokenizer, max_length, train_dataset.label_encoder)
    test_dataset = DomainDataset(test_path, tokenizer, max_length, train_dataset.label_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader, test_loader, train_dataset.num_classes, train_dataset.label_encoder, vocab_size

