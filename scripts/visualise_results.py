import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

sys.path.append(sys.path[0] + "/..")
from src.db_connector import get_db_engine, fetch_logs_batch
from src.pipeline import get_text_embedding


def plot_purity_heatmap(df):
    """
    Generates a Heatmap: Actual Source vs. Predicted Cluster
    """
    print("Generating Purity Heatmap...")

    # 1. Create a "Ground Truth" label (Source + Level)
    df["truth"] = df["source"] + " [" + df["level"] + "]"

    # 2. Create a Cross-Tabulation (Count of logs per Truth/Cluster pair)
    # We select top 20 clusters to keep the chart readable
    top_clusters = df["cluster_id"].value_counts().head(20).index
    df_filtered = df[df["cluster_id"].isin(top_clusters)]

    confusion_matrix = pd.crosstab(df_filtered["truth"], df_filtered["cluster_id"])

    # 3. Plot
    plt.figure(figsize=(14, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Cluster Purity Heatmap (Truth vs. Cluster ID)")
    plt.ylabel("True Log Source")
    plt.xlabel("Assigned Cluster ID")

    # Save
    plt.tight_layout()
    plt.savefig("models/heatmap_purity.png")
    print("Saved to models/heatmap_purity.png")


def plot_tsne_clusters(df):
    """
    Generates a 2D Scatter Plot of the high-dimensional vectors.
    """
    print("Generating t-SNE Scatter Plot (This takes ~30-60s)...")

    # 1. Generate Vectors (since we don't store them in 'logs' table)
    # Limit to 1000 points for speed
    df_sample = df.head(1000).copy()

    embeddings = []
    for _, row in df_sample.iterrows():
        txt = f"{row['message']} {row['parsed_data']}"
        embeddings.append(get_text_embedding(txt))

    X = np.array(embeddings)

    # 2. Run t-SNE (384 dims -> 2 dims)
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto"
    )
    X_embedded = tsne.fit_transform(X)

    # 3. Plot
    plt.figure(figsize=(12, 8))

    # Color by Cluster ID
    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=df_sample["cluster_id"],
        palette="tab10",
        legend="full",
        alpha=0.7,
    )

    plt.title("t-SNE Visualization of Log Clusters")
    plt.savefig("models/tsne_clusters.png")
    print("Saved to models/tsne_clusters.png")


def main():
    engine = get_db_engine()

    # Fetch data that has been clustered
    query = "SELECT * FROM logs WHERE cluster_id IS NOT NULL LIMIT 2000;"
    df = fetch_logs_batch(engine, query)

    if df.empty:
        print("No clustered data found.")
        return

    # Run Visualizations
    plot_purity_heatmap(df)
    plot_tsne_clusters(df)


if __name__ == "__main__":
    main()
