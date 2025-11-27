import torch
import os

class Config:
    # Auto-detect dataset path based on current working directory
    if os.path.exists('dataset/UIT-VSFC-train.json'):
        # Running from Lab-3 directory
        TRAIN_PATH = 'dataset/UIT-VSFC-train.json'
        DEV_PATH = 'dataset/UIT-VSFC-dev.json'
        TEST_PATH = 'dataset/UIT-VSFC-test.json'
    else:
        # Running from module_bai_2 directory
        TRAIN_PATH = '../dataset/UIT-VSFC-train.json'
        DEV_PATH = '../dataset/UIT-VSFC-dev.json'
        TEST_PATH = '../dataset/UIT-VSFC-test.json'
    
    LABELS = ['training_program', 'lecturer', 'others', 'facility']
    NUM_CLASSES = len(LABELS)
    LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
    
    # Model hyperparameters
    NUM_LAYERS = 5
    HIDDEN_SIZE = 256
    EMBEDDING_DIM = 300
    DROPOUT = 0.3
    BIDIRECTIONAL = True
    
    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 20
    MAX_SEQ_LENGTH = 128
    
    # LR scheduler
    USE_SCHEDULER = True
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2
    
    # Vocabulary
    MIN_FREQ = 2
    MAX_VOCAB_SIZE = 20000
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    
    # Device and training
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    PATIENCE = 5

