import torch
import torch.nn as nn

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        query = hidden[-1].unsqueeze(2)
        keys = self.attn(encoder_outputs)
        scores = torch.bmm(keys, query).squeeze(2)
        scores.masked_fill_(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = LuongAttention(hidden_size)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        context, attn_weights = self.attention(hidden, encoder_outputs, mask)
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)
        combined = torch.cat((output, context), dim=1)
        prediction = self.fc(combined)
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        return (src != 0).to(self.device)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size, device=self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        mask = self.create_mask(src)
        input = tgt[:, 0]
        for t in range(1, max_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
        return outputs

    def translate(self, src, max_length, sos_idx, eos_idx):
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            mask = self.create_mask(src)
            batch_size = src.shape[0]
            input = torch.full((batch_size,), sos_idx, device=self.device, dtype=torch.long)
            translations = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for _ in range(max_length):
                if finished.all():
                    break
                output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
                top1 = output.argmax(1)
                translations.append(top1.unsqueeze(1))
                finished |= top1.eq(eos_idx)
                input = top1
            if len(translations) == 0:
                return torch.full((batch_size, 1), eos_idx, device=self.device, dtype=torch.long)
            return torch.cat(translations, dim=1)

