import argparse
import sys
import os

from config import Config
from vocab import Vocabulary
from data_loader import get_data_loaders
from model import EncoderDecoderNER
from train import train_model
from evaluate import evaluate_model
from utils import set_seed, print_metrics, plot_training_history, save_history

def parse_args():
    parser = argparse.ArgumentParser(description='Encoder-Decoder LSTM for NER')
    
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--encoder_layers', type=int, help='Number of encoder LSTM layers')
    parser.add_argument('--decoder_layers', type=int, help='Number of decoder LSTM layers')
    parser.add_argument('--hidden_size', type=int, help='Hidden size')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--no_scheduler', action='store_true', help='Disable LR scheduler')
    
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()
    
    if args.dropout is not None:
        config.DROPOUT = args.dropout
    if args.encoder_layers is not None:
        config.ENCODER_NUM_LAYERS = args.encoder_layers
    if args.decoder_layers is not None:
        config.DECODER_NUM_LAYERS = args.decoder_layers
    if args.hidden_size is not None:
        config.HIDDEN_SIZE = args.hidden_size
    if args.embedding_dim is not None:
        config.EMBEDDING_DIM = args.embedding_dim
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
    
    print(f'\n{"="*60}')
    print('CONFIGURATION')
    print(f'{"="*60}')
    print(f'Model: Encoder-Decoder LSTM')
    print(f'Encoder: {config.ENCODER_NUM_LAYERS} layers | Decoder: {config.DECODER_NUM_LAYERS} layers')
    print(f'Hidden size: {config.HIDDEN_SIZE} | Embedding: {config.EMBEDDING_DIM}')
    print(f'Dropout: {config.DROPOUT} | Bidirectional: {config.BIDIRECTIONAL}')
    print(f'Batch size: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}')
    print(f'Weight decay: {config.WEIGHT_DECAY} | Epochs: {config.NUM_EPOCHS}')
    print(f'Scheduler: {config.USE_SCHEDULER} | Patience: {config.PATIENCE}')
    print(f'Device: {config.DEVICE}')
    print(f'{"="*60}\n')
    
    set_seed(config.SEED)
    
    print('[INFO] Building vocabulary...')
    sentences = []
    for path in [config.TRAIN_PATH]:
        with open(path, 'r', encoding='utf-8') as f:
            sentence = []
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        sentence.append(parts[0])
                else:
                    if sentence:
                        sentences.append(' '.join(sentence))
                        sentence = []
            if sentence:
                sentences.append(' '.join(sentence))
    
    vocab = Vocabulary(pad_token=config.PAD_TOKEN, unk_token=config.UNK_TOKEN)
    vocab.build_vocab(sentences, min_freq=config.MIN_FREQ, max_vocab_size=config.MAX_VOCAB_SIZE)
    
    print('\n[INFO] Loading datasets...')
    train_loader, dev_loader, test_loader = get_data_loaders(vocab, config)
    
    print('\n[INFO] Initializing model...')
    model = EncoderDecoderNER(len(vocab), config)
    print(f'Model parameters: {model.count_parameters():,}')
    
    print('\n[INFO] Training model...')
    model, history = train_model(model, train_loader, dev_loader, config)
    
    print('\n[INFO] Saving training history...')
    history_img_path = 'module_bai_3/training_history.png'
    history_json_path = 'module_bai_3/training_history.json'
    plot_training_history(history, save_path=history_img_path)
    save_history(history, save_path=history_json_path)
    
    print('\n[INFO] Evaluating on test set...')
    test_metrics, _, _ = evaluate_model(model, test_loader, config.DEVICE, config)
    print_metrics(test_metrics, prefix='Test')
    
    print('\n[INFO] Complete!')
    print(f'To display plot: %run module_bai_3/display_results.py')

if __name__ == '__main__':
    main()

