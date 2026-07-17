def test_no_cross_split_leakage(train_df, test_df):
    overlap = set(train_df.case_id) & set(test_df.case_id)
    assert len(overlap) == 0, f"{len(overlap)} case_ids leaked across the split"
