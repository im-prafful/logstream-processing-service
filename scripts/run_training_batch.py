import sys
import pandas as pd

# Allow imports from project root
sys.path.append(sys.path[0] + "/..")

from src.db_connector import (
    get_db_engine,
    fetch_logs_batch,
    save_embedding,
)
from src.pipeline import (
    get_text_embedding,
    build_feature_dict,
    create_streaming_pipeline,
)
from src.model import create_new_model, save_model


def main():
    print("Connecting to database...")
    engine = get_db_engine()

    # Fetch initial batch for model bootstrapping
    query = """
        SELECT *
        FROM logs
        WHERE level IN ('warning','error')
        ORDER BY log_id ASC
        LIMIT 2000;
    """

    df_logs = fetch_logs_batch(engine, query)

    if df_logs.empty:
        print("No logs found. Cannot train model.")
        return

    print(f"Loaded {len(df_logs)} logs. Beginning initial model training...")

    # Create fresh model + fresh pipeline
    model = create_new_model()
    pipeline = create_streaming_pipeline()  # important: unfrozen, learns encoders

    for _, log in df_logs.iterrows():
        log_id = log["log_id"]
        app_id = log["app_id"]
        level = log["level"]
        source = log["source"]

        # Combine text + parsed data for richer embeddings
        full_text = f"{log['message']}. Parsed Context: {log['parsed_data']}"

        # Step 1: Generate embedding
        embedding_vector = get_text_embedding(full_text)

        # Step 2: Build full feature dictionary
        feature_dict = build_feature_dict(level, source, embedding_vector)

        # Step 3: Train pipeline (encoders/scalers)
        pipeline.learn_one(feature_dict)
        processed_features = pipeline.transform_one(feature_dict)

        # Step 4: Train DenStream
        model.learn_one(processed_features)

        # Step 5: Predict cluster
        cluster_id = model.predict_one(processed_features)

        # Save embedding + assigned cluster
        save_embedding(
            engine=engine,
            log_id=log_id,
            app_id=app_id,
            embedding_vector=embedding_vector,
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

    print("Saving trained pipeline and model to disk...")
    save_model(model, pipeline)

    print("Training batch complete. Model and pipeline saved.")


if __name__ == "__main__":
    main()
