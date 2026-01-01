import torch
from .config import Config
from .data_loader import get_data_loaders
from .transformer import DomainClassifier
from .trainer import Trainer
from .visualizer import plot_training_history, plot_metrics_through_epochs, create_metrics_table, display_metrics_table

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, dev_loader, test_loader, num_classes, label_encoder, vocab_size = get_data_loaders(
        Config.DATA['train_path'],
        Config.DATA['dev_path'],
        Config.DATA['test_path'],
        batch_size=Config.DATA_LOADER['batch_size'],
        max_length=Config.DATA_LOADER['max_length'],
        min_freq=Config.DATA_LOADER['min_freq'],
        config=Config
    )
    
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {label_encoder.classes_}')
    
    model = DomainClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=Config.MODEL['d_model'],
        num_layers=Config.MODEL['num_layers'],
        num_heads=Config.MODEL['num_heads'],
        d_ff=Config.MODEL['d_ff'],
        max_len=Config.MODEL['max_len'],
        dropout=Config.MODEL['dropout']
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        learning_rate=Config.TRAINING['learning_rate'],
        num_epochs=Config.TRAINING['num_epochs'],
        patience=Config.TRAINING['patience']
    )
    
    history = trainer.train()
    
    test_metrics = trainer.evaluate(test_loader)
    
    print("\nTest Metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    
    fig1 = plot_training_history(history)
    
    fig2 = plot_metrics_through_epochs(history)
    
    df, summary_df = create_metrics_table(history, test_metrics)
    display_metrics_table(df, summary_df)

if __name__ == '__main__':
    main()

