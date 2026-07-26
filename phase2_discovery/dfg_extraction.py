"""
Directly-follows graph extraction via pm4py — produces the exact
shared/process_graph.json schema Person B's shielding logic consumes.
"""
import json
import pm4py
import pandas as pd


def extract_dfg(train_df: pd.DataFrame, case_id_col="case_id", activity_col="activity", timestamp_col="timestamp"):
    event_log = pm4py.format_dataframe(
        train_df, case_id=case_id_col, activity_key=activity_col, timestamp_key=timestamp_col
    )
    dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
    return dfg, start_activities, end_activities


def dfg_to_shared_schema(dfg: dict, train_df: pd.DataFrame, activity_col="activity", timestamp_col="timestamp") -> dict:
    durations = {}
    for case_id, group in train_df.sort_values(timestamp_col).groupby("case_id"):
        acts = group[activity_col].tolist()
        times = group[timestamp_col].tolist()
        for i in range(len(acts) - 1):
            key = (acts[i], acts[i + 1])
            dur = (times[i + 1] - times[i]).total_seconds()
            durations.setdefault(key, []).append(dur)

    graph = {}
    for (a, b), freq in dfg.items():
        avg_dur = sum(durations.get((a, b), [0])) / max(len(durations.get((a, b), [1])), 1)
        graph.setdefault(a, []).append({
            "next_activity": b,
            "frequency": int(freq),
            "avg_duration": round(avg_dur, 1),
        })
    return graph


def save_process_graph(graph: dict, path: str = "shared/process_graph.json"):
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)


if __name__ == "__main__":
    from phase1_ingestion.temporal_split import temporal_split
    import pm4py as _pm4py

    log = _pm4py.read_xes("data/BPI Challenge 2017.xes")
    df = _pm4py.convert_to_dataframe(log)
    df = df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp",
    })

    train_df, test_df = temporal_split(df)
    dfg, starts, ends = extract_dfg(train_df)
    graph = dfg_to_shared_schema(dfg, train_df)
    save_process_graph(graph)

    print(f"Discovered {len(graph)} activity nodes, {sum(len(v) for v in graph.values())} edges")
    print("✅ Saved to shared/process_graph.json — this REPLACES Person B's synthetic placeholder")
