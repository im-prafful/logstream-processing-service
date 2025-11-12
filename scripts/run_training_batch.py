import sys
import pandas as pd

# Allow imports from project root
sys.path.append(sys.path[0] + "/..")

from src.db_connector import (
    get_db_engine,
    fetch_logs_batch,
    save_embedding,
)
from src.pipeline import get_text_embedding, create_streaming_pipeline
from src.model import create_new_model, save_model


def main():
    print("Connecting to database...")
    engine = get_db_engine()

    # Fetch initial batch of logs for training
    query = "SELECT * FROM logs where level in ('warning','error') ORDER BY log_id ASC LIMIT 2000;"
    df_logs = fetch_logs_batch(engine, query)

    if df_logs.empty:
        print("No logs found. Cannot train model.")
        return

    print(f"Loaded {len(df_logs)} logs. Beginning initial model training...")

    # Create a fresh pipeline and model for first-time training
    pipeline = create_streaming_pipeline()
    model = create_new_model()

    for _, log in df_logs.iterrows():
        log_id = log["log_id"]
        app_id = log["app_id"]
        level = log["level"]
        source = log["source"]
        message = log["message"]

        # Step 1: Create text embedding (384-D vector)
        embedding_vector = get_text_embedding(message)

        # Step 2: Build ML feature dict (structured + embedding)
        feature_dict = {"level": level, "source": source}

        # Add embedding dimensions to feature dict
        for index, emb_value in enumerate(embedding_vector):
            feature_name = f"vec_{index}"
            feature_dict[feature_name] = emb_value

        # Step 3: Train the pipeline and transform the record
        pipeline.learn_one(feature_dict)
        processed_features = pipeline.transform_one(feature_dict)

        # Step 4: Train the DenStream model incrementally
        model.learn_one(processed_features)

        # Optionally: assign micro-cluster ID
        cluster_id = model.predict_one(processed_features)

        # Save embedding + cluster to DB
        save_embedding(
            engine=engine,
            log_id=log_id,
            app_id=app_id,
            embedding_vector=embedding_vector,
            cluster_id=cluster_id,
            level=level,
            source=source,
        )

    print("Initial streaming training completed.")
    print(f"Model has {len(model.p_micro_clusters)} micro-clusters.")

    print("Saving trained pipeline and model to disk...")
    save_model(model, pipeline)

    print("Training batch complete. Model and pipeline saved.")


if __name__ == "__main__":
    main()
