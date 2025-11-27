import torch

class Config:
    TRAIN_PATH = 'dataset/UIT-VSFC-train.json'
    DEV_PATH = 'dataset/UIT-VSFC-dev.json'
    TEST_PATH = 'dataset/UIT-VSFC-test.json'

    LABELS = ['training_program', 'lecturer', 'others', 'facility']
    NUM_CLASSES = len(LABELS)
    LABEL2ID =  {label: idx for idx, label in enumerate(LABELS)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

    # Tham số mô hình
    NUM_LAYERS = 5 # Số lớp mô hình LSTM
    HIDDEN_SIZE = 256 # Kích thước mô hình
    EMBEDDING_DIM = 300 # Kích thước embedding
    DROPOUT = 0 # Tỷ lệ % hidden unit bị tắt để chống overfitting
    BIDIRECTIONAL = True # Mô hình 2 chiều

    # Tham số huấn luyện
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    MAX_SEQ_LENGTH = 128 # Độ dài tối đa của câu

    # Tham số Vocab
    MIN_FREQ = 2 # Tần suất tối thiểu để từ được giữ lại
    MAX_VOCAB_SIZE = 20000

    # Token đặc biệt
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Early stopping 
    PATIENCE = 5

if __name__ == "__main__":
    config = Config()
    print("---KIỂM TRA CẤU HÌNH---")
    print(f"Device: {config.DEVICE}")
    print(f"Số lớp LSTM: {config.NUM_LAYERS}")
    print(f"Hidden size: {config.HIDDEN_SIZE}")
    print(f"Embedding dim: {config.EMBEDDING_DIM}")
    print(f"Số classes: {config.NUM_CLASSES}")
    print(f"Labels: {config.LABELS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Max sequence length: {config.MAX_SEQ_LENGTH}")
    print(f"Bidirectional model: {config.BIDIRECTIONAL}")
    print(f"% Drop out: {config.DROPOUT}")