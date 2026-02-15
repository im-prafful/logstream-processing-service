import sys
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score

sys.path.append(sys.path[0] + "/..")
from src.db import get_db_engine, fetch_logs_batch
from src.ml import get_text_embedding


def calculate_purity(df):
    """
    Checks if clusters contain a single dominant type of log.
    Ground Truth is defined as: "Source + Level"
    """
    # Create a "True Label" based on your known generation logic
    df["true_label"] = df["source"] + "_" + df["level"]

    # metrics.homogeneity_score:
    # Checks if each cluster contains only members of a single class.
    h_score = homogeneity_score(df["true_label"], df["cluster_id"])

    # metrics.completeness_score:
    # Checks if all members of a given class are assigned to the same cluster.
    c_score = completeness_score(df["true_label"], df["cluster_id"])

    return h_score, c_score


def calculate_math_quality(df):
    """
    Calculates Silhouette Score.
    +1.0: Dense, well-separated clusters (Excellent).
    0.0: Overlapping clusters (Confused).
    -1.0: Misclustered points (Bad).
    """
    # We need the vectors to calculate distance.
    # Since we didn't save vectors in the 'logs' table (only 'log_embeddings'),
    # we regenerate them quickly or join tables.
    # For validation speed, let's regenerate or fetch from log_embeddings if available.

    # Assuming we fetch from log_embeddings for speed (if you populated it)
    # If not, we regenerate:
    embeddings = []
    print("Generating embeddings for validation (this might take a moment)...")
    for _, row in df.iterrows():
        full_text = f"{row['message']}. Parsed: {row['parsed_data']}"
        embeddings.append(get_text_embedding(full_text))

    X = np.array(embeddings)

    # Silhouette requires at least 2 clusters and > 1 sample
    if len(df["cluster_id"].unique()) < 2:
        return 0.0

    return silhouette_score(X, df["cluster_id"])


def main():
    print("--- STARTING MODEL VALIDATION ---")
    engine = get_db_engine()

    # Fetch clustered logs
    # We only care about logs that HAVE a cluster_id
    query = """
        SELECT * FROM logs 
        WHERE cluster_id IS NOT NULL 
        LIMIT 2000;
    """
    df = fetch_logs_batch(engine, query)

    if df.empty:
        print("No clustered logs found. Run incremental batch first.")
        return

    print(f"Auditing {len(df)} logs...")

    # 1. Functional Validation (Purity)
    h_score, c_score = calculate_purity(df)
    print(f"\n FUNCTIONAL METRICS (Vs. Ground Truth):")
    print(f"   Homogeneity:  {h_score:.2f} / 1.0  (Are clusters pure?)")
    print(f"   Completeness: {c_score:.2f} / 1.0  (Are error types fragmented?)")

    if h_score > 0.8:
        print("   PASSED: Clusters are highly pure.")
    elif h_score > 0.5:
        print("   WARNING: Some clusters are mixed.")
    else:
        print("   FAILED: Clusters are messy.")

    # 2. Mathematical Validation (Silhouette)
    # Only run this if you have time/compute, it's O(N^2) complexity
    s_score = calculate_math_quality(df)
    print(f"\n MATHEMATICAL METRICS (Geometry):")
    print(f"   Silhouette Score: {s_score:.2f} (Range: -1 to 1)")

    if s_score > 0.4:
        print("   PASSED: Clusters are distinct and dense.")
    elif s_score > 0.1:
        print("   WARNING: Clusters are somewhat overlapping.")
    else:
        print("   FAILED: Clusters are indistinguishable blobs.")

    # 3. "Eyeball" Test (Show examples)
    print("\n👀 CLUSTER SAMPLES:")
    top_clusters = df["cluster_id"].value_counts().head(3).index.tolist()

    for cid in top_clusters:
        print(f"\n   [Cluster {cid}]")
        sample = df[df["cluster_id"] == cid].head(3)
        for _, row in sample.iterrows():
            print(f"     - [{row['source']}] {row['message'][:60]}...")


if __name__ == "__main__":
    main()
