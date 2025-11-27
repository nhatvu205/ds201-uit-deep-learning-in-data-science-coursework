import json
import os
import argparse
import torch.nn as nn

from config import Config
from vocab import Vocabulary
from data_loader import get_data_loaders
from model import LSTMClassifier
from train import train_model
from evaluate import evaluate_model
from utils import print_metrics, plot_training_history


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LSTM Topic Classification - UIT-VSFC')
    
    # Model hyperparameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate (default: from config)')
    parser.add_argument('--num_layers', type=int, default=None, help='Number of LSTM layers (default: from config)')
    parser.add_argument('--hidden_size', type=int, default=None, help='Hidden size (default: from config)')
    parser.add_argument('--embedding_dim', type=int, default=None, help='Embedding dimension (default: from config)')
    parser.add_argument('--bidirectional', type=bool, default=None, help='Use bidirectional LSTM (default: from config)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: from config)')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate (default: from config)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs (default: from config)')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (default: from config)')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override config with command line arguments
    if args.dropout is not None:
        config.DROPOUT = args.dropout
    if args.num_layers is not None:
        config.NUM_LAYERS = args.num_layers
    if args.hidden_size is not None:
        config.HIDDEN_SIZE = args.hidden_size
    if args.embedding_dim is not None:
        config.EMBEDDING_DIM = args.embedding_dim
    if args.bidirectional is not None:
        config.BIDIRECTIONAL = args.bidirectional
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.num_epochs is not None:
        config.NUM_EPOCHS = args.num_epochs
    if args.patience is not None:
        config.PATIENCE = args.patience
    
    # Print configuration
    print("="*60)
    print("CẤU HÌNH MÔ HÌNH")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Dropout: {config.DROPOUT}")
    print(f"Số lớp LSTM: {config.NUM_LAYERS}")
    print(f"Hidden size: {config.HIDDEN_SIZE}")
    print(f"Embedding dim: {config.EMBEDDING_DIM}")
    print(f"Bidirectional: {config.BIDIRECTIONAL}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Số epochs: {config.NUM_EPOCHS}")
    print(f"Patience: {config.PATIENCE}")
    print(f"Max sequence length: {config.MAX_SEQ_LENGTH}")
    print(f"Số classes: {config.NUM_CLASSES}")
    print(f"Labels: {config.LABELS}")
    print("="*60)
    
    # Build vocabulary
    print("\n[INFO] Xây dựng vocabulary...")
    with open(config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        sentences = [item['sentence'] for item in train_data]
    
    vocab = Vocabulary(pad_token=config.PAD_TOKEN, unk_token=config.UNK_TOKEN)
    vocab.build_vocab(sentences, min_freq=config.MIN_FREQ, max_vocab_size=config.MAX_VOCAB_SIZE)
    
    # Load datasets
    print("\n[INFO] Load dataset")
    train_loader, dev_loader, test_loader = get_data_loaders(vocab, config)
    
    # Initialize model
    print("\n[INFO] Khởi tạo mô hình")
    model = LSTMClassifier(len(vocab), config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Số tham số: {num_params:,}")
    
    # Train model
    print("\n[INFO] Huấn luyện mô hình")
    model, history = train_model(model, train_loader, dev_loader, config)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n[INFO] Đánh giá trên test set")
    crit = nn.CrossEntropyLoss()
    test_loss, test_metrics, test_labels, test_preds = evaluate_model(model, test_loader, crit, config.DEVICE)
    print_metrics(test_metrics, prefix="Test")
    
    print("\nHoàn tất!")


if __name__ == "__main__":
    main()
