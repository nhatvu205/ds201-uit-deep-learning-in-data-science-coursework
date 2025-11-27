import argparse
import json
import os
import sys

from config import Config
from vocab import Vocabulary
from data_loader import get_data_loaders
from model import GRUClassifier
from train import train_model
from evaluate import evaluate_model
from utils import set_seed, print_metrics, plot_training_history, save_history
def parse_args():
    parser = argparse.ArgumentParser(description='GRU Topic Classification')
    
    # Model hyperparameters
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, help='Number of GRU layers')
    parser.add_argument('--hidden_size', type=int, help='Hidden size')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional GRU')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 regularization)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    
    # LR scheduler
    parser.add_argument('--no_scheduler', action='store_true', help='Disable LR scheduler')
    parser.add_argument('--scheduler_factor', type=float, help='LR scheduler factor')
    parser.add_argument('--scheduler_patience', type=int, help='LR scheduler patience')
    
    # Vocabulary
    parser.add_argument('--min_freq', type=int, help='Minimum word frequency')
    parser.add_argument('--max_vocab_size', type=int, help='Maximum vocabulary size')
    parser.add_argument('--max_seq_length', type=int, help='Maximum sequence length')
    
    return parser.parse_args()

def main():
    args = parse_args()
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
    if args.bidirectional:
        config.BIDIRECTIONAL = True
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.weight_decay is not None:
        config.WEIGHT_DECAY = args.weight_decay
    if args.num_epochs is not None:
        config.NUM_EPOCHS = args.num_epochs
    if args.patience is not None:
        config.PATIENCE = args.patience
    if args.no_scheduler:
        config.USE_SCHEDULER = False
    if args.scheduler_factor is not None:
        config.SCHEDULER_FACTOR = args.scheduler_factor
    if args.scheduler_patience is not None:
        config.SCHEDULER_PATIENCE = args.scheduler_patience
    if args.min_freq is not None:
        config.MIN_FREQ = args.min_freq
    if args.max_vocab_size is not None:
        config.MAX_VOCAB_SIZE = args.max_vocab_size
    if args.max_seq_length is not None:
        config.MAX_SEQ_LENGTH = args.max_seq_length
    
    # Print configuration
    print('CONFIGURATION')
    print(f'{"="*60}')
    print(f'Model: GRU with {config.NUM_LAYERS} layers')
    print(f'Hidden size: {config.HIDDEN_SIZE} | Embedding: {config.EMBEDDING_DIM}')
    print(f'Dropout: {config.DROPOUT} | Bidirectional: {config.BIDIRECTIONAL}')
    print(f'Batch size: {config.BATCH_SIZE} | Learning rate: {config.LEARNING_RATE}')
    print(f'Weight decay: {config.WEIGHT_DECAY} | Epochs: {config.NUM_EPOCHS}')
    print(f'Scheduler: {config.USE_SCHEDULER} | Patience: {config.PATIENCE}')
    print(f'Device: {config.DEVICE}')
    print(f'{"="*60}\n')
    
    set_seed(config.SEED)
    
    # Build vocabulary
    print('[INFO] Building vocabulary...')
    with open(config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    sentences = [item['sentence'] for item in train_data]
    
    vocab = Vocabulary(pad_token=config.PAD_TOKEN, unk_token=config.UNK_TOKEN)
    vocab.build_vocab(sentences, min_freq=config.MIN_FREQ, max_vocab_size=config.MAX_VOCAB_SIZE)
    
    # Load datasets
    print('\n[INFO] Loading datasets...')
    train_loader, dev_loader, test_loader = get_data_loaders(vocab, config)
    
    # Initialize model
    print('\n[INFO] Initializing model...')
    model = GRUClassifier(len(vocab), config)
    print(f'Model parameters: {model.count_parameters():,}')
    
    # Train model
    print('\n[INFO] Training model...')
    model, history = train_model(model, train_loader, dev_loader, config)
    
    # Save and plot training history
    print('\n[INFO] Saving training history...')
    history_img_path = 'module_bai_2/training_history.png'
    history_json_path = 'module_bai_2/training_history.json'
    plot_training_history(history, save_path=history_img_path)
    save_history(history, save_path=history_json_path)
    
    # Evaluate on test set
    print('\n[INFO] Evaluating on test set...')
    test_metrics, _, _ = evaluate_model(model, test_loader, config.DEVICE)
    print_metrics(test_metrics, prefix='Test')
    
    print('\n[INFO] Complete!')

if __name__ == '__main__':
    main()

