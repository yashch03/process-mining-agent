"""
Latency profiling for the deviation-scoring function.
Section 7.3: must meet a <=50ms fixed latency budget, reported as a
full distribution (p50/p95/p99), not a single point claim.
"""
import time
import numpy as np
from phase3_evaluation.fusion import deviation_score


def profile_latency(n_trials: int = 1000) -> dict:
    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        deviation_score(nll=1.5, graph_cost=0.8)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # convert to ms

    latencies = np.array(latencies)
    return {
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "max_ms": latencies.max(),
    }


if __name__ == "__main__":
    results = profile_latency()
    print(f"p50 latency: {results['p50_ms']:.4f} ms")
    print(f"p95 latency: {results['p95_ms']:.4f} ms")
    print(f"p99 latency: {results['p99_ms']:.4f} ms")
    print(f"max latency: {results['max_ms']:.4f} ms")

    budget_ms = 50
    if results["p99_ms"] <= budget_ms:
        print(f"✅ p99 latency ({results['p99_ms']:.4f} ms) is within the {budget_ms}ms budget")
    else:
        print(f"⚠️  p99 latency ({results['p99_ms']:.4f} ms) EXCEEDS the {budget_ms}ms budget")
