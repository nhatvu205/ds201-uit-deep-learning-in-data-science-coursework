import torch
from torch.utils.data import DataLoader, random_split
import os

from mnist_dataset import MnistDataset
from lenet_model import LeNet
from train import train_model
from evaluate import evaluate_model, print_evaluation_results, plot_confusion_matrix, plot_training_history

def main():
    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset IDX paths (trong thư mục dataset/)
    train_image_path = 'dataset/train-images.idx3-ubyte'
    train_label_path = 'dataset/train-labels.idx1-ubyte'
    test_image_path = 'dataset/t10k-images.idx3-ubyte'
    test_label_path = 'dataset/t10k-labels.idx1-ubyte'

    # Load training dataset
    full_train_dataset = MnistDataset(train_image_path, train_label_path)
    
    # Split train and validation
    train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load test dataset
    test_dataset = MnistDataset(test_image_path, test_label_path)
    print(f"Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=MnistDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=MnistDataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=MnistDataset.collate_fn
    )
    
    # Initialize model
    model = LeNet()
    
    # Training
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # Evaluation
    metrics, predictions, labels = evaluate_model(model, test_loader, device)
    print_evaluation_results(metrics)
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(labels, predictions)

if __name__ == '__main__':
    main()
