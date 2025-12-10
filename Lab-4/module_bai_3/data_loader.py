import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from .vocabulary import Vocabulary

class TranslationDataset(Dataset):
    def __init__(self, data_path, src_vocab, tgt_vocab, max_length, sample_size=None):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        if sample_size is not None and sample_size < len(self.data):
            random.seed(42)
            self.data = random.sample(self.data, sample_size)

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
    train_dataset = TranslationDataset(config.train_path, src_vocab, tgt_vocab, config.max_length, config.train_sample_size)
    dev_dataset = TranslationDataset(config.dev_path, src_vocab, tgt_vocab, config.max_length, config.dev_sample_size)
    test_dataset = TranslationDataset(config.test_path, src_vocab, tgt_vocab, config.max_length, config.test_sample_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    return train_loader, dev_loader, test_loader

def build_vocabularies(config):
    with open(config.train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    if config.train_sample_size is not None and config.train_sample_size < len(train_data):
        random.seed(42)
        train_data = random.sample(train_data, config.train_sample_size)
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_sentences = [item['english'].lower().strip() for item in train_data]
    tgt_sentences = [item['vietnamese'].lower().strip() for item in train_data]
    src_vocab.build_from_sentences(src_sentences)
    tgt_vocab.build_from_sentences(tgt_sentences)
    return src_vocab, tgt_vocab

