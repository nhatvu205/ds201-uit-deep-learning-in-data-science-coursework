import torch
import torch.nn as nn
from config import Config

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, config):
        super(GRUClassifier, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=0)
        
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL
        )
        
        self.dropout = nn.Dropout(config.DROPOUT)
        
        gru_output_size = config.HIDDEN_SIZE * 2 if config.BIDIRECTIONAL else config.HIDDEN_SIZE
        self.fc = nn.Linear(gru_output_size, config.NUM_CLASSES)
    
    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, hidden = self.gru(packed_embedded)
        
        if self.config.BIDIRECTIONAL:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

