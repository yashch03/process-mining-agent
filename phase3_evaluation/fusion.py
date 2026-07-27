"""
Deviation scoring: fuses sequence-model NLL with graph-conformance cost.
Section 7.1 of the implementation doc.
"""
import numpy as np


def deviation_score(nll: float, graph_cost: float, lam: float = 0.3) -> float:
    """
    Fusion function combining negative log-likelihood from the sequence model
    and the token-replay alignment cost from the discovered process graph.
    """
    return lam * nll + (1 - lam) * graph_cost


def batch_deviation_scores(nlls: np.ndarray, graph_costs: np.ndarray, lam: float = 0.3) -> np.ndarray:
    return lam * nlls + (1 - lam) * graph_costs


if __name__ == "__main__":
    conforming_nll, conforming_cost = 0.5, 0.0
    deviant_nll, deviant_cost = 3.2, 2.0

    conforming_score = deviation_score(conforming_nll, conforming_cost)
    deviant_score = deviation_score(deviant_nll, deviant_cost)

    print(f"Conforming trace deviation score: {conforming_score:.3f}")
    print(f"Deviant trace deviation score: {deviant_score:.3f}")
    assert deviant_score > conforming_score, "Deviant trace should score higher"
    print("✅ Fusion function behaves correctly — deviant traces score higher")
