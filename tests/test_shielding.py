import pytest
from phase4_agent.shielding.graph_verifier import shield_score, load_process_graph


@pytest.fixture
def sample_graph():
    return load_process_graph("shared/process_graph.json")


@pytest.fixture
def valid_sequence():
    # Real BPI-2017 transition, frequency 25145 in the discovered graph
    return ["A_Accepted", "O_Create Offer"]


@pytest.fixture
def invalid_sequence():
    # Never observed as a direct transition in the discovered graph
    return ["A_Accepted", "A_Cancelled"]


def test_shield_score_zero_for_valid_sequence(sample_graph, valid_sequence):
    assert shield_score(valid_sequence, sample_graph) == 0


def test_shield_score_positive_for_invalid_sequence(sample_graph, invalid_sequence):
    assert shield_score(invalid_sequence, sample_graph) > 0
