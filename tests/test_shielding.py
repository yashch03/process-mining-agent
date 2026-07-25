def test_shield_score_zero_for_valid_sequence(sample_graph, valid_sequence):
    from phase4_agent.shielding.graph_verifier import shield_score
    assert shield_score(valid_sequence, sample_graph) == 0
