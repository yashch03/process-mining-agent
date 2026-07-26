import pandas as pd
import pytest
from phase1_ingestion.temporal_split import temporal_split, assert_no_leakage


@pytest.fixture
def sample_events():
    return pd.DataFrame({
        "case_id": ["A", "A", "B", "B", "C"],
        "activity": ["submit", "approve", "submit", "reject", "submit"],
        "timestamp": pd.to_datetime([
            "2016-10-01", "2016-10-15",
            "2016-10-20", "2016-11-05",
            "2016-11-10",
        ]),
    })


def test_no_cross_split_leakage(sample_events):
    train_df, test_df = temporal_split(sample_events, split_date="2016-11-01")
    assert_no_leakage(train_df, test_df)


def test_case_b_stays_whole_in_train(sample_events):
    train_df, test_df = temporal_split(sample_events, split_date="2016-11-01")
    assert "B" in train_df["case_id"].values
    assert "B" not in test_df["case_id"].values
