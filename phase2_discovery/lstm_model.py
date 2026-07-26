"""
LSTM next-event predictor — reimplementing Tax et al. (2017).
"""
import torch
import torch.nn as nn


class NextEventLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.activity_head = nn.Linear(hidden_dim, vocab_size)
        self.time_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embed(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        activity_logits = self.activity_head(last_hidden)
        time_pred = self.time_head(last_hidden)
        return activity_logits, time_pred


def build_vocab(train_df, activity_col="activity") -> dict:
    activities = sorted(train_df[activity_col].unique())
    return {act: i for i, act in enumerate(activities)}


if __name__ == "__main__":
    vocab_size = 26
    model = NextEventLSTM(vocab_size=vocab_size)
    dummy_input = torch.randint(0, vocab_size, (4, 10))
    activity_logits, time_pred = model(dummy_input)
    print("Activity logits shape:", activity_logits.shape)
    print("Time prediction shape:", time_pred.shape)
    print("LSTM architecture runs correctly")
