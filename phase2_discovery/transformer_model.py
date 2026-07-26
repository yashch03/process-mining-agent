"""
Transformer next-event predictor — benchmarked against the LSTM baseline
to evaluate self-attention advantages on process event sequences.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class NextEventTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.activity_head = nn.Linear(d_model, vocab_size)
        self.time_head = nn.Linear(d_model, 1)

    def forward(self, x):
        embedded = self.embed(x)
        embedded = self.pos_encoding(embedded)
        encoded = self.encoder(embedded)
        last_hidden = encoded[:, -1, :]
        activity_logits = self.activity_head(last_hidden)
        time_pred = self.time_head(last_hidden)
        return activity_logits, time_pred


if __name__ == "__main__":
    vocab_size = 26
    model = NextEventTransformer(vocab_size=vocab_size)
    dummy_input = torch.randint(0, vocab_size, (4, 10))
    activity_logits, time_pred = model(dummy_input)
    print("Activity logits shape:", activity_logits.shape)
    print("Time prediction shape:", time_pred.shape)
    print("Transformer architecture runs correctly")
