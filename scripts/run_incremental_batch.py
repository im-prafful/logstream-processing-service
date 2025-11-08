import sys
import pandas as pd

sys.path.append(sys.path[0] + "/..")

from src.db_connector import get_db_engine, fetch_logs_batch
from src.pipeline import get_text_embedding
from src.model import load_model, save_model


def main():
    model, pipeline = load_model()
    if pipeline is None:
        print("Error: Pipeline not found. Run 'run_training_batch.py' first.")
        return

    engine = get_db_engine()
    query = "SELECT * FROM logs WHERE cluster_id IS NULL;"
    df_new = fetch_logs_batch(engine, query)

    if df_new.empty:
        print("No new logs to process.")
        return

    print(f"Streaming {len(df_new)} new logs into model...")
    cluster_results = {}

    for _, log in df_new.iterrows():
        parsed_data = log["parsed_data"] if log["parsed_data"] else {}
        embedding = get_text_embedding(log["message"])

        x = {
            "level": log["level"],
            "source": log["source"],
            "method": parsed_data.get("method", None),
        }
        x.update({f"vec_{i}": emb_val for i, emb_val in enumerate(embedding)})

        processed_x = pipeline.learn_one(x).transform_one(x)
        model.learn_one(processed_x)
        cluster_id = model.predict_one(processed_x)

        cluster_results[log["log_id"]] = cluster_id

    print("Incremental batch complete.")

    # Have to update the 'cluster_id' in database

    save_model(model, pipeline)


if __name__ == "__main__":
    main()
