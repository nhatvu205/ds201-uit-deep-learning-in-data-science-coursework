import torch
import os

class Config:
    if os.path.exists('PhoNER_COVID19/data/word'):
        TRAIN_PATH = 'PhoNER_COVID19/data/word/train_word.conll'
        DEV_PATH = 'PhoNER_COVID19/data/word/dev_word.conll'
        TEST_PATH = 'PhoNER_COVID19/data/word/test_word.conll'
    else:
        TRAIN_PATH = '../PhoNER_COVID19/data/word/train_word.conll'
        DEV_PATH = '../PhoNER_COVID19/data/word/dev_word.conll'
        TEST_PATH = '../PhoNER_COVID19/data/word/test_word.conll'
    
    # NER tags for PhoNER_COVID19
    TAGS = ['O', 'B-PATIENT_ID', 'I-PATIENT_ID', 'B-NAME', 'I-NAME', 
            'B-AGE', 'I-AGE', 'B-GENDER', 'I-GENDER', 'B-JOB', 'I-JOB',
            'B-LOCATION', 'I-LOCATION', 'B-ORGANIZATION', 'I-ORGANIZATION',
            'B-SYMPTOM_AND_DISEASE', 'I-SYMPTOM_AND_DISEASE', 
            'B-TRANSPORTATION', 'I-TRANSPORTATION', 'B-DATE', 'I-DATE']
    
    NUM_TAGS = len(TAGS)
    TAG2ID = {tag: idx for idx, tag in enumerate(TAGS)}
    ID2TAG = {idx: tag for idx, tag in enumerate(TAGS)}
    
    # Model hyperparameters
    ENCODER_NUM_LAYERS = 5
    DECODER_NUM_LAYERS = 5
    HIDDEN_SIZE = 256
    EMBEDDING_DIM = 300
    DROPOUT = 0.3
    BIDIRECTIONAL = True
    
    # Training hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 30
    MAX_SEQ_LENGTH = 256
    
    # LR scheduler
    USE_SCHEDULER = True
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    
    # Vocabulary
    MIN_FREQ = 2
    MAX_VOCAB_SIZE = 30000
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    PATIENCE = 5

