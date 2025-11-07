import sys
import pandas as pd

sys.path.append(sys.path[0] + "/..")

from src.db_connector import get_db_engine, fetch_logs_batch
from src.pipeline import create_streaming_pipeline, get_text_embedding
from src.model import create_new_model, save_model


def main():
    engine = get_db_engine()
    df = fetch_logs_batch(engine, "SELECT * FROM logs LIMIT 1000;")
    if df.empty:
        print("No data found to train on.")
        return

    pipeline = create_streaming_pipeline()
    model = create_new_model()

    print("Streaming initial batch into model...")
    cluster_results = {}

    for _, log in df.iterrows():
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

        # cluster_id = model.predict_one(processed_x)
        # cluster_results[log["log_id"]] = cluster_id

    print("Initial training complete.")
    print(f"Model now has {len(model.micro_clusters)} micro-clusters.")

    save_model(model, pipeline)


if __name__ == "__main__":
    main()
