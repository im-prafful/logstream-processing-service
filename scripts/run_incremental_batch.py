from sqlalchemy import text  # Fixed import
import sys
import os

sys.path.append(sys.path[0] + "/..")

from src.db_connector import (
    detect_and_create_incidents,
    get_db_engine,
    fetch_logs_batch,
    save_embedding,
    save_pattern,
    fetch_min_timestamp,
)
from src.vector_engine import SemanticVectorEngine
from src.pipeline import get_text_embedding, build_feature_dict
from src.model import load_model

PRODUCTION_DIR = "models/production"


def main():
    print("--- STARTING INCREMENTAL INFERENCE (BLUE Deployment) ---")

    # 1. ALWAYS LOAD FROM PRODUCTION
    # If a swap happened recently, this will pick up the new file automatically.
    model, pipeline = load_model(directory=PRODUCTION_DIR)

    if model is None:
        print("Waiting for initial training to complete...")
        return

    # Load Vector Engine from Production
    vector_engine = SemanticVectorEngine(minkowski_p=1.5, threshold=0.35)
    vector_path = os.path.join(PRODUCTION_DIR, "vector_centroids.pkl")
    vector_engine.load(vector_path)

    engine = get_db_engine()

    # 2. PROCESS NEW LOGS (Read-Only Inference)
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
    global_timestamp = fetch_min_timestamp(
        engine=engine, timestamp_query=timestamp_query
    )

    if df_new.empty:
        print("No new logs.")
        return

    batch_size = len(df_new)  # Track batch size
    print(f"Classifying {batch_size} logs using LIVE model...")

    for _, log in df_new.iterrows():
        full_text = f"{log['message']}. Parsed: {log['parsed_data']}"
        embedding = get_text_embedding(full_text)
        sem_id = vector_engine.get_semantic_group(embedding, log["log_id"])
        feats = build_feature_dict(log["level"], log["source"], embedding, sem_id)

        proc_feats = pipeline.transform_one(feats)

        # PREDICT ONLY (No learn_one)
        cluster_id = model.predict_one(proc_feats)

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

    detect_and_create_incidents(
        engine=engine, batch_size=batch_size, global_timestamp=global_timestamp
    )  # Pass batch_size

    print("Inference batch complete.")


if __name__ == "__main__":
    main()
