"""
Temporal hold-out splitting — the leakage-prevention core of Phase 1.
Splits by absolute event timestamp, then verifies no case_id spans both splits.
"""
import pandas as pd


def temporal_split(
    events_df: pd.DataFrame,
    split_date: str = "2016-11-01",
    case_id_col: str = "case_id",
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split events by absolute timestamp — NOT random/hash-based, to avoid
    leaking future process behavior into training.

    A case that starts before split_date but has events after it is assigned
    WHOLLY to train, based on its earliest event — this is the correct way
    to prevent partial-case leakage across the boundary.
    """
    split_dt = pd.Timestamp(split_date)

    # Match timezone-awareness of the actual data — BPI-2017 timestamps are UTC-aware
    case_start = events_df.groupby(case_id_col)[timestamp_col].min()
    if case_start.dt.tz is not None and split_dt.tz is None:
        split_dt = split_dt.tz_localize(case_start.dt.tz)

    train_case_ids = case_start[case_start < split_dt].index
    test_case_ids = case_start[case_start >= split_dt].index

    train_df = events_df[events_df[case_id_col].isin(train_case_ids)].copy()
    test_df = events_df[events_df[case_id_col].isin(test_case_ids)].copy()

    return train_df, test_df


def assert_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, case_id_col: str = "case_id") -> None:
    """Raises AssertionError if any case_id appears in both splits."""
    overlap = set(train_df[case_id_col]) & set(test_df[case_id_col])
    assert len(overlap) == 0, f"{len(overlap)} case_ids leaked across the split: {list(overlap)[:5]}"


if __name__ == "__main__":
    import pm4py

    log = pm4py.read_xes("data/BPI Challenge 2017.xes")
    df = pm4py.convert_to_dataframe(log)

    df = df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp",
        "org:resource": "resource",
    })

    train_df, test_df = temporal_split(df)
    assert_no_leakage(train_df, test_df)

    print(f"Train: {len(train_df)} events, {train_df['case_id'].nunique()} cases")
    print(f"Test:  {len(test_df)} events, {test_df['case_id'].nunique()} cases")
    print("✅ No cross-split leakage")
