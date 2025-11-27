from collections import Counter
from typing import List

class Vocabulary:
    def __init__(self, pad_token='<PAD>', unk_token='<UNK>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.word_count = Counter()
    
    def build_vocab(self, sentences: List[str], min_freq: int = 2, max_vocab_size: int = None):
        print('[INFO] Building vocabulary...')
        for sentence in sentences:
            words = sentence.lower().strip().split()
            self.word_count.update(words)
        
        if max_vocab_size:
            most_common = self.word_count.most_common(max_vocab_size - 2)
        else:
            most_common = self.word_count.most_common()
        
        for word, count in most_common:
            if count >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f'[INFO] Vocab size: {len(self.word2idx)}')
    
    def encode(self, sentence: str) -> List[int]:
        words = sentence.lower().strip().split()
        return [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
    
    def decode(self, indices: List[int]) -> str:
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)

