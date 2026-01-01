import torch
import torch.nn as nn
import sys
import os

from config import Config
from data_loader import build_vocabularies, create_dataloaders
from model import Encoder, Decoder, Seq2Seq
from train import train_model
from evaluate import evaluate_model, show_translation_examples, plot_training_history, plot_metric_history

def main():
    print("Initializing configuration...")
    config = Config()
    print(config)
    
    print("\n[INFO] Building vocabularies from training data...")
    src_vocab, tgt_vocab = build_vocabularies(config)
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    print("\n[INFO] Creating data loaders...")
    train_loader, dev_loader, test_loader = create_dataloaders(config, src_vocab, tgt_vocab)
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\n[INFO] Building model...")
    encoder = Encoder(
        vocab_size=len(src_vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    decoder = Decoder(
        vocab_size=len(tgt_vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    model = Seq2Seq(encoder, decoder, config.device).to(config.device)
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    param.data[m.hidden_size:m.hidden_size*2].fill_(1)
    
    model.apply(init_weights)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total trainable parameters: {total_params:,}")
    
    print("\nTraining model...")
    train_losses, val_losses, val_rouge = train_model(model, train_loader, dev_loader, config, tgt_vocab)
    print("\nTraining completed!")
    
    print("\n[INFO] Displaying training history...")
    plot_training_history(train_losses, val_losses)
    if len(val_rouge) > 0:
        plot_metric_history(val_rouge, metric_name='ROUGE-L')
    
    print("\n[INFO] Evaluating on test set...")
    test_results = evaluate_model(model, test_loader, config, tgt_vocab)
    
    show_translation_examples(model, test_loader, config, src_vocab, tgt_vocab, num_examples=5)
    
    print("\n[INFO] Completed!")

if __name__ == '__main__':
    main()

