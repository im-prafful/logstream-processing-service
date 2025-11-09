import sys
import pandas as pd

# Allow importing from project root
sys.path.append(sys.path[0] + "/..")

from src.db_connector import get_db_engine, fetch_logs_batch, save_embedding
from src.pipeline import get_text_embedding
from src.model import load_model, save_model


def main():
    print("Loading model and pipeline...")
    model, pipeline = load_model()

    if pipeline is None:
        print(
            "ERROR: Preprocessing pipeline not found. Run 'run_training_batch.py' first."
        )
        return

    engine = get_db_engine()

    # Fetch logs that haven't yet been processed and stored in log_embeddings
    query = """
        SELECT *
        FROM logs
        WHERE log_id NOT IN (SELECT log_id FROM log_embeddings)
        ORDER BY log_id ASC
        LIMIT 2000;
    """
    df_new = fetch_logs_batch(engine, query)

    if df_new.empty:
        print("No new logs found. Everything is already processed.")
        return

    print(f"🚀 Processing {len(df_new)} new logs...")

    for _, log in df_new.iterrows():
        log_id = log["log_id"]
        app_id = log["app_id"]
        message = log["message"]
        level = log["level"]
        source = log["source"]

        # Step 1: Encode the text into a 384-dimensional vector
        embedding_vector = get_text_embedding(message)

        # Step 2: Build the feature dictionary for the pipeline (level, source, parsed_data)
        feature_dict = {"level": level, "source": source}

        # Add each vector component individually to the feature dict
        for index, emb_val in enumerate(embedding_vector):
            feature_name = f"vec_{index}"
            feature_dict[feature_name] = emb_val

        # Step 3: Update the pipeline (fit encoders/scalers) and transform the data
        pipeline.learn_one(feature_dict)
        processed_features = pipeline.transform_one(feature_dict)

        # Step 4: Incrementally train the clustering model with this log
        model.learn_one(processed_features)

        # Step 5: Predict which cluster this log belongs to
        cluster_id = model.predict_one(processed_features)

        # Step 6: Save embedding + metadata into the database
        save_embedding(
            engine=engine,
            log_id=log_id,
            app_id=app_id,
            embedding_vector=embedding_vector,
            cluster_id=cluster_id,
            level=level,
            source=source,
        )

    print("Finished processing all new logs.")
    print("Saving updated model and pipeline...")
    save_model(model, pipeline)
    print("All done! Model and pipeline updated.")


if __name__ == "__main__":
    main()
