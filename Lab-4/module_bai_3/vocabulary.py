import json
from collections import Counter
import pickle

class Vocabulary:
    def __init__(self, max_vocab_size=50000):
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.max_vocab_size = max_vocab_size
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.SOS_TOKEN)
        self.add_word(self.EOS_TOKEN)
        self.add_word(self.UNK_TOKEN)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1

    def build_from_sentences(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                self.word_count[word] += 1
        for word, _ in self.word_count.most_common(self.max_vocab_size - 4):
            if word not in self.word2idx:
                self.add_word(word)

    def encode(self, sentence, max_length):
        words = sentence.split()
        indices = [self.word2idx.get(w, self.word2idx[self.UNK_TOKEN]) for w in words]
        indices = [self.word2idx[self.SOS_TOKEN]] + indices + [self.word2idx[self.EOS_TOKEN]]
        if len(indices) > max_length:
            indices = indices[:max_length - 1] + [self.word2idx[self.EOS_TOKEN]]
        else:
            indices += [self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices))
        return indices

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx[self.EOS_TOKEN]:
                break
            if idx not in (self.word2idx[self.PAD_TOKEN], self.word2idx[self.SOS_TOKEN]):
                words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return ' '.join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

