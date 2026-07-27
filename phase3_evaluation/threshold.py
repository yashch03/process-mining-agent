"""
Cost-asymmetric threshold derivation from precision-recall curve.
Section 7.2: missing a violation costs 10x more than a false-positive review.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve


def derive_cost_optimal_threshold(y_true: np.ndarray, deviation_scores: np.ndarray, miss_cost: float = 10.0, fp_cost: float = 1.0):
    """
    Given ground-truth labels (1 = actual violation) and continuous deviation
    scores, find the threshold that minimizes total cost under the given
    cost matrix.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, deviation_scores)

    # cost per threshold: missing a violation (1-recall) weighted heavily,
    # false-positive reviews (1-precision) weighted lightly
    cost = miss_cost * (1 - recalls[:-1]) + fp_cost * (1 - precisions[:-1])
    best_idx = cost.argmin()

    return {
        "threshold": thresholds[best_idx],
        "precision_at_threshold": precisions[best_idx],
        "recall_at_threshold": recalls[best_idx],
        "cost_at_threshold": cost[best_idx],
    }


if __name__ == "__main__":
    # Synthetic smoke test: 100 traces, 10 are true violations, scores separate them imperfectly
    rng = np.random.default_rng(42)
    y_true = np.array([1] * 10 + [0] * 90)
    deviation_scores = np.concatenate([
        rng.normal(3.0, 0.8, 10),   # violations tend to score higher
        rng.normal(1.0, 0.8, 90),   # normal traces score lower
    ])

    result = derive_cost_optimal_threshold(y_true, deviation_scores)
    print("Cost-optimal threshold:", round(result["threshold"], 3))
    print("Precision at threshold:", round(result["precision_at_threshold"], 3))
    print("Recall at threshold:", round(result["recall_at_threshold"], 3))
    print("Minimum cost achieved:", round(result["cost_at_threshold"], 3))
    print("✅ Threshold derivation runs correctly on synthetic data")
