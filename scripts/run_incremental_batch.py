from sqlalchemy import text
import sys
import os
import time

# 1. Force logs to flush immediately (fixes the "missing logs" issue)
sys.stdout.reconfigure(line_buffering=True)

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
    # READ ENV VARIABLES SENT BY LAMBDA
    batch_id = os.environ.get("BATCH_ID")
    start_log_id = os.environ.get("START_LOG_ID")
    end_log_id = os.environ.get("END_LOG_ID")
    db_host = os.environ.get("DB_HOST") 

    print(f"--- STARTING BATCH {batch_id} (Logs {start_log_id} - {end_log_id}) ---")

    # Safety Check: If run locally without Env Vars, warn the user
    if not batch_id or not start_log_id:
        print("❌ ERROR: Missing Batch ID or Log Range environment variables.")
        return

    # 1. LOAD MODEL
    model, pipeline = load_model(directory=PRODUCTION_DIR)
    if model is None:
        print("Waiting for initial training to complete...")
        return

    # Load Vector Engine
    vector_engine = SemanticVectorEngine(minkowski_p=1.5, threshold=0.35)
    vector_path = os.path.join(PRODUCTION_DIR, "vector_centroids.pkl")
    vector_engine.load(vector_path)

    engine = get_db_engine()

    # 2. PROCESS SPECIFIC BATCH (The Logic Change)
    # We remove the "LIMIT 2000" and instead use the precise range from Lambda
    print(f"Fetching logs between ID {start_log_id} and {end_log_id}...")
    
    query = f"""
        SELECT * FROM logs 
        WHERE log_id BETWEEN {start_log_id} AND {end_log_id}
        AND level IN ('error','warning')
        AND cluster_id IS NULL 
        ORDER BY log_id ASC
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
        print(f"Batch {batch_id} is empty (No error/warning logs found in range).")
        # Still need to mark as complete in DB?
        return

    batch_size = len(df_new)
    print(f"Classifying {batch_size} logs for Batch {batch_id}...")

    for _, log in df_new.iterrows():
        full_text = f"{log['message']}. Parsed: {log['parsed_data']}"
        embedding = get_text_embedding(full_text)
        sem_id = vector_engine.get_semantic_group(embedding, log["log_id"])
        feats = build_feature_dict(log["level"], log["source"], embedding, sem_id)

        proc_feats = pipeline.transform_one(feats)
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
    detect_and_create_incidents(engine=engine, batch_size=batch_size, global_timestamp=global_timestamp)

    # 3. CRITICAL: Mark Batch as COMPLETED in DB
    # The Lambda launched us and forgot about us. WE must close the loop.
    with engine.connect() as conn:
        print(f"Marking Batch {batch_id} as COMPLETED in Database...")
        conn.execute(text(f"UPDATE batch_order SET status='COMPLETED', processed_at=NOW() WHERE batchid={batch_id}"))
        conn.commit()

    print(f" Batch {batch_id} execution finished successfully.")

if __name__ == "__main__":
    main()