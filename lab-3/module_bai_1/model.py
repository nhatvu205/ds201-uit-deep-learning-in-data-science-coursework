import torch
import torch.nn as nn
from config import Config

class LSTMClassifier(nn.Module):
    
    def __init__(self, vocab_size, config):
        super(LSTMClassifier, self).__init__()
        
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL
        )

        self.dropout = nn.Dropout(config.DROPOUT)

        lstm_output_size = config.HIDDEN_SIZE *2 if config.BIDIRECTIONAL else config.HIDDEN_SIZE

        self.fc = nn.Linear(lstm_output_size, config.NUM_CLASSES)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    
        if self.config.BIDIRECTIONAL:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        hidden = self.dropout(hidden)

        logits = self.fc(hidden)

        return logits

    def predict(self, input_ids, lengths):
        logits = self.forward(input_ids, lengths)
        predictions = torch.argmax(logits, dim=1)
        return predictions
