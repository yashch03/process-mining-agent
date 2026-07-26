"""
Training loop for LSTM and Transformer next-event predictors.
Logs to W&B, checkpoints per epoch, uses seeds from configs/seeds.yaml.
"""
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

from phase1_ingestion.temporal_split import temporal_split
from phase2_discovery.lstm_model import NextEventLSTM, build_vocab
from phase2_discovery.transformer_model import NextEventTransformer


class NextEventDataset(Dataset):
    """Sliding-window next-event prediction dataset."""
    def __init__(self, df, vocab, case_id_col="case_id", activity_col="activity", timestamp_col="timestamp", max_len=10):
        self.samples = []
        for case_id, group in df.sort_values(timestamp_col).groupby(case_id_col):
            acts = [vocab[a] for a in group[activity_col] if a in vocab]
            for i in range(1, len(acts)):
                seq = acts[max(0, i - max_len):i]
                seq = [0] * (max_len - len(seq)) + seq
                target = acts[i]
                self.samples.append((seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train_one_model(model_class, model_name, vocab_size, train_loader, test_loader, seed, epochs, device, project="process-mining-agent"):
    torch.manual_seed(seed)
    model = model_class(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    run = wandb.init(project=project, name=f"{model_name}_seed{seed}", reinit=True,
                      config={"model": model_name, "seed": seed, "epochs": epochs})

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, target in train_loader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            logits, _ = model(seq)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seq, target in test_loader:
                seq, target = seq.to(device), target.to(device)
                logits, _ = model(seq)
                preds = logits.argmax(dim=-1)
                correct += (preds == target).sum().item()
                total += target.size(0)
        test_acc = correct / total

        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "test_accuracy": test_acc})
        print(f"[{model_name} seed={seed}] epoch {epoch}: train_loss={avg_train_loss:.4f}, test_acc={test_acc:.4f}")

    run.finish()
    return test_acc


if __name__ == "__main__":
    import pm4py

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    with open("configs/seeds.yaml") as f:
        seeds_cfg = yaml.safe_load(f)

    src = "/kaggle/input/datasets/yashch03/bpi-challenge-2017/BPI Challenge 2017.xes"
    log = pm4py.read_xes(src)
    df = pm4py.convert_to_dataframe(log)
    df = df.rename(columns={
        "case:concept:name": "case_id", "concept:name": "activity", "time:timestamp": "timestamp",
    })

    train_df, test_df = temporal_split(df)
    vocab = build_vocab(train_df)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    train_dataset = NextEventDataset(train_df, vocab)
    test_dataset = NextEventDataset(test_df, vocab)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    seed = seeds_cfg["lstm_seeds"][0]
    epochs = 2

    lstm_acc = train_one_model(NextEventLSTM, "LSTM", vocab_size, train_loader, test_loader, seed, epochs, device)
    transformer_acc = train_one_model(NextEventTransformer, "Transformer", vocab_size, train_loader, test_loader, seed, epochs, device)

    print(f"\nLSTM test accuracy: {lstm_acc:.4f}")
    print(f"Transformer test accuracy: {transformer_acc:.4f}")
    print(f"Markov baseline was: 0.667")
