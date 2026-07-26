"""
Trace vectorization + SVD dimensionality reduction + k-means clustering
to isolate major process variants (Section 6.1).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


def vectorize_traces(train_df: pd.DataFrame, case_id_col="case_id", activity_col="activity", timestamp_col="timestamp"):
    """
    Bag-of-activities vectorization: each trace becomes a vector of
    activity counts. Simple, interpretable, sufficient for variant clustering.
    """
    activities = sorted(train_df[activity_col].unique())
    act_to_idx = {a: i for i, a in enumerate(activities)}

    case_ids = []
    vectors = []
    for case_id, group in train_df.sort_values(timestamp_col).groupby(case_id_col):
        vec = np.zeros(len(activities))
        for act in group[activity_col]:
            vec[act_to_idx[act]] += 1
        case_ids.append(case_id)
        vectors.append(vec)

    return np.array(vectors), case_ids, activities


def reduce_and_cluster(trace_vectors: np.ndarray, n_components: int = 10, n_clusters: int = 6, random_state: int = 42):
    """SVD dimensionality reduction, then k-means to isolate process variants."""
    n_components = min(n_components, trace_vectors.shape[1] - 1, trace_vectors.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(trace_vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(reduced)

    return clusters, reduced, svd.explained_variance_ratio_.sum()


if __name__ == "__main__":
    from phase1_ingestion.temporal_split import temporal_split
    import pm4py

    log = pm4py.read_xes("data/BPI Challenge 2017.xes")
    df = pm4py.convert_to_dataframe(log)
    df = df.rename(columns={
        "case:concept:name": "case_id", "concept:name": "activity", "time:timestamp": "timestamp",
    })

    train_df, test_df = temporal_split(df)
    trace_vectors, case_ids, activities = vectorize_traces(train_df)

    clusters, reduced, explained_var = reduce_and_cluster(trace_vectors)

    print(f"Vectorized {len(case_ids)} traces across {len(activities)} activities")
    print(f"SVD explained variance ratio: {explained_var:.3f}")
    print(f"Cluster sizes: {np.bincount(clusters)}")
    print("SVD + k-means clustering runs correctly")
