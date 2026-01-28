from sqlalchemy import text
import sys
import os
import pandas as pd

sys.path.append(sys.path[0] + "/..")

from src.db_connector import (
    get_db_engine,
    fetch_logs_batch,
    save_embedding,
    save_pattern,
    fetch_min_timestamp,
    save_cluster_stats,
    fetch_cluster_history,
    create_incident,
)
from src.volume_analyzer import VolumeAnomalyDetector
from src.vector_engine import SemanticVectorEngine
from src.pipeline import get_text_embedding, build_feature_dict
from src.model import load_model

PRODUCTION_DIR = "models/production"


def main():
    print("--- STARTING INCREMENTAL INFERENCE (BLUE Deployment) ---")

    # 1. ALWAYS LOAD FROM PRODUCTION
    model, pipeline = load_model(directory=PRODUCTION_DIR)

    if model is None:
        print("Waiting for initial training to complete...")
        return

    # Load Vector Engine
    vector_engine = SemanticVectorEngine(minkowski_p=1.5, threshold=0.35)
    vector_path = os.path.join(PRODUCTION_DIR, "vector_centroids.pkl")
    vector_engine.load(vector_path)

    engine = get_db_engine()

    # 2. PROCESS NEW LOGS
    query = """
        SELECT * FROM logs 
        WHERE log_id NOT IN (SELECT log_id FROM log_embeddings)
        AND level IN ('error','warning') 
        LIMIT 2000;
    """

    timestamp_query = text(
        """
        SELECT MIN(timestamp) 
        FROM logs 
        WHERE level IN ('error','warning') 
        AND cluster_id IS NULL
    """
    )

    df_new = fetch_logs_batch(engine, query)

    # We don't strictly need global_timestamp for volume analysis anymore,
    global_timestamp = fetch_min_timestamp(
        engine=engine, timestamp_query=timestamp_query
    )

    if df_new.empty:
        print("No new logs.")
        return

    batch_size = len(df_new)
    print(f"Classifying {batch_size} logs using LIVE model...")

    batch_counts = {}

    for _, log in df_new.iterrows():
        full_text = f"{log['message']}. Parsed: {log['parsed_data']}"
        embedding = get_text_embedding(full_text)
        sem_id = vector_engine.get_semantic_group(embedding, log["log_id"])
        feats = build_feature_dict(log["level"], log["source"], embedding, sem_id)

        proc_feats = pipeline.transform_one(feats)

        cluster_id = model.predict_one(proc_feats)

        # Update the count for this cluster
        if cluster_id not in batch_counts:
            batch_counts[cluster_id] = 0
        batch_counts[cluster_id] += 1

        save_embedding(
            engine,
            log["log_id"],
            log["app_id"],
            embedding,
            cluster_id,
            log["level"],
            log["source"],
        )

    save_pattern(engine=engine)

    # VOLUME ANALYSIS LOGIC

    # 1.

    # 2. SAVE TO HISTORY
    # Now 'batch_counts' actually exists and contains data
    save_cluster_stats(engine, batch_counts)

    # 3. LOAD THE VOLUME MODEL
    vol_model = VolumeAnomalyDetector(window_size=5)
    vol_model.load(PRODUCTION_DIR)

    # 4. FETCH CONTEXT
    history_df = fetch_cluster_history(engine, window_size=5)

    # 5. RUN INFERENCE
    anomalies = vol_model.detect_anomalies(history_df)

    if anomalies:
        print(f"🚨 DETECTED VOLUME ANOMALIES in Clusters: {anomalies}")
        for cid in anomalies:
            create_incident(engine, cid, reason="Abnormal Volume Spike")
    else:
        print("✅ No volume anomalies detected in this batch.")

    print("Inference batch complete.")


if __name__ == "__main__":
    main()
