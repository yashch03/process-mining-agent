"""
Process-graph shielding: scores candidate action sequences against
the discovered process graph. Lower score = more conforming.
"""
import json
from pathlib import Path


def load_process_graph(path: str = "shared/process_graph.json") -> dict:
    with open(path) as f:
        return json.load(f)


def shield_score(candidate_action_sequence: list[str], process_graph: dict) -> int:
    """
    Penalize candidate sequences that deviate from the discovered DFG.
    Returns 0 for a fully conforming sequence, higher = more violations.
    """
    cost = 0
    for i in range(len(candidate_action_sequence) - 1):
        curr, nxt = candidate_action_sequence[i], candidate_action_sequence[i + 1]
        valid_next = [edge["next_activity"] for edge in process_graph.get(curr, [])]
        if nxt not in valid_next:
            cost += 1
    return cost


def rerank_candidates(candidates: list[list[str]], process_graph: dict, top_k: int = 1) -> list[tuple[list[str], int]]:
    scored = [(c, shield_score(c, process_graph)) for c in candidates]
    return sorted(scored, key=lambda x: x[1])[:top_k]


if __name__ == "__main__":
    graph = load_process_graph()

    valid_seq = ["A_Accepted", "O_Create Offer"]
    invalid_seq = ["A_Accepted", "A_Cancelled"]

    print("Valid sequence score:", shield_score(valid_seq, graph))
    print("Invalid sequence score:", shield_score(invalid_seq, graph))
