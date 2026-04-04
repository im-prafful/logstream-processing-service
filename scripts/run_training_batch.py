import pandas as pd
import sys
import os
import shutil
import csv
import json
import torch

sys.stdout.reconfigure(line_buffering=True)

sys.path.append(sys.path[0] + "/..")

from src.db import (
    get_db_engine,
    fetch_logs_batch,
    save_embedding,
    save_pattern,
)
from src.ml import (
    SemanticVectorEngine,
    build_feature_dict,
    create_streaming_pipeline,
    create_new_model,
    save_model,
    VolumeAnomalyDetector,
)
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

# CONSTANTS FOR BLUE/GREEN DEPLOYMENT
PRODUCTION_DIR = "scripts/models/production"
STAGING_DIR = "models/staging"

# Temporary CSV file written during the training loop.
# Acts as a crash-resilient staging buffer before the final DB insert.
STAGING_CSV = "staging/embeddings_staging.csv"

# ── GPU / CPU Auto-Detection ──────────────────────────────────────────────────
# Uses your NVIDIA RTX 3050 (CUDA) when running locally.
# Falls back to CPU gracefully if CUDA is not available.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(
    f"[DEVICE] Using: {device.upper()}  |  CUDA available: {torch.cuda.is_available()}"
)
if device == "cuda":
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")

# Load the embedding model once, bound to the detected device
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def get_text_embedding_local(text: str):
    """Single-item encode — kept for compatibility but not used in training loop."""
    return embedding_model.encode(text, convert_to_numpy=True)


def batch_encode_texts(texts: list, batch_size: int = 64):
    """
    Encode all texts in one GPU-batched call.
    Returns a list of numpy arrays (one embedding per text).
    batch_size=64 fits comfortably in RTX 3050 4GB VRAM.
    """
    print(
        f"[EMBED] Encoding {len(texts)} texts on {device.upper()} (batch_size={batch_size})..."
    )
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"[EMBED] Done. Embedding shape: {embeddings.shape}")
    return embeddings


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

    # ── OPTIMISATION 1: Pre-compute ALL embeddings in a single GPU-batched call ──
    # Instead of calling encode() 5,000 times inside the loop, we build the full
    # text list up front and let the GPU crunch them all at once.
    print("Pre-computing embeddings for all logs (GPU-batched)...")
    all_texts = [
        f"{row['message']}. Parsed: {row['parsed_data']}"
        for _, row in df_logs.iterrows()
    ]
    all_embeddings = batch_encode_texts(all_texts, batch_size=64)

    # 2. TRAIN NEW MODEL (isolated in memory/staging)
    print("Training Base Model...")
    vector_engine = SemanticVectorEngine(minkowski_p=1.5, threshold=0.35)
    model = create_new_model()
    pipeline = create_streaming_pipeline()

    # ── OPTIMISATION 2: Write each row to a CSV staging file during the loop ──────
    # Instead of one DB transaction per log (5,000 round-trips), we stream rows
    # to a local CSV file as the loop runs. If the script crashes mid-way, the
    # file already has everything processed so far — nothing is lost.
    os.makedirs(os.path.dirname(STAGING_CSV), exist_ok=True)
    CSV_COLUMNS = ["log_id", "app_id", "embedding", "cluster_id", "level", "source"]

    print(f"[CSV] Streaming rows to staging file: {STAGING_CSV}")
    with open(STAGING_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for idx, (_, log) in enumerate(df_logs.iterrows()):
            log_id = log["log_id"]
            app_id = log["app_id"]
            level = log["level"]
            source = log["source"]

            # Use the pre-computed embedding for this row
            embedding = all_embeddings[idx]

            sem_id = vector_engine.get_semantic_group(embedding, log_id)
            feats = build_feature_dict(log["level"], log["source"], embedding, sem_id)

            pipeline.learn_one(feats)
            proc_feats = pipeline.transform_one(feats)
            model.learn_one(proc_feats)
            cluster_id = model.predict_one(proc_feats)

            # Write this row to the CSV immediately (crash-safe)
            writer.writerow(
                {
                    "log_id": log_id,
                    "app_id": app_id,
                    "embedding": json.dumps(
                        embedding.tolist()
                    ),  # store vector as JSON string
                    "cluster_id": cluster_id,
                    "level": level,
                    "source": source,
                }
            )

    print(f"[CSV] Finished writing staging file.")

    # ── SINGLE BULK INSERT: read the CSV, one transaction, one round-trip ─────────
    print(f"[DB] Reading staging file and bulk inserting into log_embeddings...")
    df_staging = pd.read_csv(STAGING_CSV)

    # Deserialise the embedding JSON strings back into Python lists
    embedding_rows = [
        {
            "log_id": row["log_id"],
            "app_id": row["app_id"],
            "embedding": json.loads(row["embedding"]),
            "cluster_id": row["cluster_id"],
            "level": row["level"],
            "source": row["source"],
        }
        for _, row in df_staging.iterrows()
    ]
    cluster_updates = [
        {"log_id": row["log_id"], "cluster_id": row["cluster_id"]}
        for _, row in df_staging.iterrows()
    ]

    insert_embeddings_sql = text(
        """
        INSERT INTO log_embeddings (log_id, app_id, embedding, cluster_id, level, source)
        VALUES (:log_id, :app_id, :embedding, :cluster_id, :level, :source)
        ON CONFLICT (log_id) DO NOTHING;
    """
    )
    update_logs_sql = text(
        """
        UPDATE logs SET cluster_id = :cluster_id WHERE log_id = :log_id;
    """
    )

    with engine.begin() as conn:
        conn.execute(insert_embeddings_sql, embedding_rows)
        conn.execute(update_logs_sql, cluster_updates)
    print(f"[DB] Bulk insert complete ({len(embedding_rows)} rows).")

    # Clean up the staging file now that it's safely in the DB
    os.remove(STAGING_CSV)
    print(f"[CSV] Staging file cleaned up.")

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

    # 4. TRAIN VOLUME ANOMALY MODEL
    print("Training Volume Analysis Model...")

    # 4A. SIMULATE BATCHES
    # We split the training data into small virtual batches, creating a "Time Series" history from our static data.
    df_logs["virtual_batch_id"] = df_logs.index // 100

    # 4B. COUNT LOGS PER CLUSTER PER VIRTUAL BATCH
    # We query the DB to get the cluster_ids we just assigned during the loop above
    volume_query = """
        SELECT cluster_id, count(*) as log_count, (log_id / 100) as batch_id 
        FROM log_embeddings 
        GROUP BY cluster_id, batch_id
        ORDER BY batch_id ASC
    """
    df_volume_history = pd.read_sql(volume_query, engine)

    # Rename 'batch_id' to 'batch_timestamp' to match the analyzer's expectation
    df_volume_history = df_volume_history.rename(
        columns={"batch_id": "batch_timestamp"}
    )

    # 4C. TRAIN THE ISOLATION FOREST
    # We assume a window size of 5
    vol_model = VolumeAnomalyDetector(window_size=5)
    vol_model.train(df_volume_history)

    # 4D. SAVE TO STAGING
    vol_model.save(STAGING_DIR)

    # 5. THE ATOMIC SWAP (Blue/Green Switch)
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
