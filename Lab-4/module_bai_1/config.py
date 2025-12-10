import torch

class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.hidden_size = 256
        self.num_layers = 3
        self.embedding_dim = 256
        self.dropout = 0.3
        
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.num_epochs = 15
        self.max_length = 128
        self.early_stopping_patience = 3
        
        self.train_path = 'dataset/train.json'
        self.dev_path = 'dataset/dev.json'
        self.test_path = 'dataset/test.json'
        
        self.teacher_forcing_ratio = 0.7
        
    def __str__(self):
        config_str = "Configuration:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str

