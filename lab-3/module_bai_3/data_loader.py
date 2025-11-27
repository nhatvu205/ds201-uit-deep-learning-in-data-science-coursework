import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

from config import Config
from vocab import Vocabulary

class NERDataset(Dataset):
    def __init__(self, data_path, vocab, config):
        self.vocab = vocab
        self.config = config
        self.sentences, self.tags = self.load_data(data_path)
    
    def load_data(self, data_path):
        sentences, tags = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            sentence, tag_seq = [], []
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        word, tag = parts[0], parts[-1]
                        sentence.append(word)
                        tag_seq.append(tag)
                else:
                    if sentence:
                        sentences.append(' '.join(sentence))
                        tags.append(tag_seq)
                        sentence, tag_seq = [], []
            
            if sentence:
                sentences.append(' '.join(sentence))
                tags.append(tag_seq)
        
        print(f'[INFO] Loaded {len(sentences)} sentences from {data_path}')
        return sentences, tags
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.vocab.encode(self.sentences[idx])
        tag_ids = [self.config.TAG2ID[tag] for tag in self.tags[idx]]
        
        if len(input_ids) > self.config.MAX_SEQ_LENGTH:
            input_ids = input_ids[:self.config.MAX_SEQ_LENGTH]
            tag_ids = tag_ids[:self.config.MAX_SEQ_LENGTH]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)

def collate_fn(batch, pad_idx=0, tag_pad_idx=0):
    inputs, tags = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=tag_pad_idx)
    return padded_inputs, padded_tags, lengths

def get_data_loaders(vocab, config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = NERDataset(config.TRAIN_PATH, vocab, config)
    dev_dataset = NERDataset(config.DEV_PATH, vocab, config)
    test_dataset = NERDataset(config.TEST_PATH, vocab, config)
    pad_idx = vocab.word2idx[config.PAD_TOKEN]
    tag_pad_idx = config.TAG2ID['O']
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                             collate_fn=lambda x: collate_fn(x, pad_idx, tag_pad_idx))
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                           collate_fn=lambda x: collate_fn(x, pad_idx, tag_pad_idx))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=lambda x: collate_fn(x, pad_idx, tag_pad_idx))
    
    return train_loader, dev_loader, test_loader

