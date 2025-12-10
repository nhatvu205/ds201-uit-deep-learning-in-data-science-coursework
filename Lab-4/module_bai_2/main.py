import torch
import torch.nn as nn
from .config import Config
from .data_loader import build_vocabularies, create_dataloaders
from .model import Encoder, Decoder, Seq2Seq
from .train import train_model
from .evaluate import evaluate_model, show_translation_examples

def init_weights(m, hidden_size):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                param.data[hidden_size:2*hidden_size].fill_(1)

def build_model(config, src_vocab_size, tgt_vocab_size):
    encoder = Encoder(src_vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, config.dropout)
    decoder = Decoder(tgt_vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, config.dropout)
    model = Seq2Seq(encoder, decoder, config.device).to(config.device)
    model.apply(lambda m: init_weights(m, config.hidden_size))
    return model

def run_train_eval():
    config = Config()
    print(config)
    src_vocab, tgt_vocab = build_vocabularies(config)
    train_loader, dev_loader, test_loader = create_dataloaders(config, src_vocab, tgt_vocab)
    model = build_model(config, len(src_vocab), len(tgt_vocab))
    train_losses, val_losses = train_model(model, train_loader, dev_loader, config)
    evaluate_model(model, test_loader, config, tgt_vocab)
    show_translation_examples(model, test_loader, config, src_vocab, tgt_vocab, num_examples=5)
    return model, src_vocab, tgt_vocab, train_losses, val_losses

if __name__ == '__main__':
    run_train_eval()

