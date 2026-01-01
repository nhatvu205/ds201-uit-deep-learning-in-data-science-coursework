class Config:
    DATA = {
        'train_path': 'PhoNER_COVID19/data/syllable/train_syllable.json',
        'dev_path': 'PhoNER_COVID19/data/syllable/dev_syllable.json',
        'test_path': 'PhoNER_COVID19/data/syllable/test_syllable.json'
    }
    
    DATA_LOADER = {
        'batch_size': 16,
        'max_length': 128,
        'min_freq': 2
    }
    
    TOKENIZER = {
        'pad_token': '<PAD>',
        'unk_token': '<UNK>',
        'cls_token': '<CLS>',
        'sep_token': '<SEP>'
    }
    
    MODEL = {
        'd_model': 512,
        'num_layers': 3,
        'num_heads': 8,
        'd_ff': 2048,
        'max_len': 128,
        'dropout': 0.1
    }
    
    TRAINING = {
        'learning_rate': 2e-5,
        'num_epochs': 30,
        'patience': 5,
        'min_delta': 0.0
    }

