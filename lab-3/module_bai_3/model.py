import torch
import torch.nn as nn
from config import Config

class EncoderDecoderNER(nn.Module):
    def __init__(self, vocab_size, config):
        super(EncoderDecoderNER, self).__init__()
        self.config = config
        
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=0)
        
        # Encoder: 5 layer LSTM
        self.encoder = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.ENCODER_NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.ENCODER_NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL
        )
        
        encoder_output_size = config.HIDDEN_SIZE * 2 if config.BIDIRECTIONAL else config.HIDDEN_SIZE
        
        # Decoder: 5 layer LSTM
        self.decoder = nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.DECODER_NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.DECODER_NUM_LAYERS > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(config.HIDDEN_SIZE, config.NUM_TAGS)
    
    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(packed_embedded)
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(encoder_output, batch_first=True)
        
        if self.config.BIDIRECTIONAL:
            batch_size = encoder_hidden.size(1)
            encoder_hidden_forward = encoder_hidden[-2*self.config.DECODER_NUM_LAYERS::2, :, :]
            encoder_hidden_backward = encoder_hidden[-2*self.config.DECODER_NUM_LAYERS+1::2, :, :]
            decoder_hidden = (encoder_hidden_forward + encoder_hidden_backward) / 2
            
            encoder_cell_forward = encoder_cell[-2*self.config.DECODER_NUM_LAYERS::2, :, :]
            encoder_cell_backward = encoder_cell[-2*self.config.DECODER_NUM_LAYERS+1::2, :, :]
            decoder_cell = (encoder_cell_forward + encoder_cell_backward) / 2
        else:
            decoder_hidden = encoder_hidden[-self.config.DECODER_NUM_LAYERS:, :, :]
            decoder_cell = encoder_cell[-self.config.DECODER_NUM_LAYERS:, :, :]
        
        decoder_output, _ = self.decoder(encoder_output, (decoder_hidden, decoder_cell))
        decoder_output = self.dropout(decoder_output)
        
        logits = self.fc(decoder_output)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

