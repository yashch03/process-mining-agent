"""
First-order Markov chain baseline — the sanity check your LSTM/Transformer
must beat before either is worth trusting.
"""
from collections import defaultdict, Counter
import pandas as pd


def build_markov_baseline(train_df: pd.DataFrame, case_id_col="case_id", activity_col="activity", timestamp_col="timestamp") -> dict:
    transitions = defaultdict(Counter)
    for case_id, group in train_df.sort_values(timestamp_col).groupby(case_id_col):
        acts = group[activity_col].tolist()
        for a, b in zip(acts, acts[1:]):
            transitions[a][b] += 1
    return {a: c.most_common(1)[0][0] for a, c in transitions.items()}


def evaluate_markov_baseline(model: dict, test_df: pd.DataFrame, case_id_col="case_id", activity_col="activity", timestamp_col="timestamp") -> float:
    correct, total = 0, 0
    for case_id, group in test_df.sort_values(timestamp_col).groupby(case_id_col):
        acts = group[activity_col].tolist()
        for a, b in zip(acts, acts[1:]):
            predicted = model.get(a)
            if predicted is not None:
                total += 1
                if predicted == b:
                    correct += 1
    return correct / total if total else 0.0


if __name__ == "__main__":
    from phase1_ingestion.temporal_split import temporal_split
    import pm4py

    log = pm4py.read_xes("data/BPI Challenge 2017.xes")
    df = pm4py.convert_to_dataframe(log)
    df = df.rename(columns={
        "case:concept:name": "case_id", "concept:name": "activity", "time:timestamp": "timestamp",
    })

    train_df, test_df = temporal_split(df)
    model = build_markov_baseline(train_df)
    acc = evaluate_markov_baseline(model, test_df)
    print(f"Markov baseline next-event accuracy: {acc:.3f}")
    print("Your LSTM and Transformer must beat this to justify their complexity")
