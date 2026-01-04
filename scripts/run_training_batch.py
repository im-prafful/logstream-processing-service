import sys
import os
import shutil

sys.path.append(sys.path[0] + "/..")

from src.db_connector import (
    get_db_engine,
    fetch_logs_batch,
    save_embedding,
    save_pattern,
)
from src.vector_engine import SemanticVectorEngine
from src.pipeline import (
    get_text_embedding,
    build_feature_dict,
    create_streaming_pipeline,
)
from src.model import create_new_model, save_model

# CONSTANTS FOR BLUE/GREEN DEPLOYMENT
PRODUCTION_DIR = "models/production"
STAGING_DIR = "models/staging"


def main():
    print("--- STARTING BACKGROUND TRAINING (GREEN Deployment) ---")

    # 1. CLEAN STAGING AREA
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)
    os.makedirs(STAGING_DIR)

    engine = get_db_engine()

    # Fetch large dataset for training
    query = "SELECT * FROM logs WHERE level IN ('warning','error') ORDER BY log_id ASC LIMIT 5000;"
    df_logs = fetch_logs_batch(engine, query)

    if df_logs.empty:
        return

    # 2. TRAIN NEW MODEL (This takes time, but it's isolated in memory/staging)
    print("Training Base Model...")
    vector_engine = SemanticVectorEngine(minkowski_p=1.5, threshold=0.35)
    model = create_new_model()
    pipeline = create_streaming_pipeline()

    for _, log in df_logs.iterrows():
        log_id = log["log_id"]
        app_id = log["app_id"]
        level = log["level"]
        source = log["source"]

        full_text = f"{log['message']}. Parsed: {log['parsed_data']}"
        embedding = get_text_embedding(full_text)
        sem_id = vector_engine.get_semantic_group(embedding, log["log_id"])
        feats = build_feature_dict(log["level"], log["source"], embedding, sem_id)

        pipeline.learn_one(feats)
        proc_feats = pipeline.transform_one(feats)
        model.learn_one(proc_feats)

        cluster_id = model.predict_one(proc_feats)

        save_embedding(
            engine=engine,
            log_id=log_id,
            app_id=app_id,
            embedding_vector=embedding,
            cluster_id=cluster_id,
            level=level,
            source=source,
        )

    # Log the number of micro-clusters detected
    try:
        print(
            f"Initial streaming training completed. Model has {len(model.p_micro_clusters)} micro-clusters."
        )
    except:
        print("Initial streaming training complete.")

    # 3. SAVE TO STAGING (The "Green" Copy)
    print(f"Training complete. Saving to STAGING ({STAGING_DIR})...")
    save_model(model, pipeline, directory=STAGING_DIR)
    vector_engine.save(os.path.join(STAGING_DIR, "vector_centroids.pkl"))
    save_pattern(engine)

    # 4. THE ATOMIC SWAP (Blue/Green Switch)
    print("Performing ZERO-DOWNTIME SWAP...")

    # Strategy: Rename Production -> Backup, Rename Staging -> Production
    BACKUP_DIR = "models/backup_previous_version"

    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)

    if os.path.exists(PRODUCTION_DIR):
        os.rename(PRODUCTION_DIR, BACKUP_DIR)  # Move old live model aside

    os.rename(STAGING_DIR, PRODUCTION_DIR)  # Move new model to live slot

    print(f"✅ SWAP COMPLETE. New model is live in {PRODUCTION_DIR}")


if __name__ == "__main__":
    main()
