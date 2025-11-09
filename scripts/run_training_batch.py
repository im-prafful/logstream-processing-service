import sys
import pandas as pd

sys.path.append(sys.path[0] + "/..")

# Import helper functions from your own code.
from src.db_connector import get_db_engine, fetch_logs_batch
from src.pipeline_utils import create_streaming_pipeline
from src.embedding_utils import get_text_embedding
from src.model import create_new_model, save_model


def main():
    # Step 1: Connect to the database
    engine = get_db_engine()

    # Step 2: Fetch logs from the database (only 1000 for now)
    query = "SELECT * FROM logs LIMIT 1000;"
    df = fetch_logs_batch(engine, query)

    if df.empty:
        print("No data found to train on.")
        return


    pipeline = create_streaming_pipeline()
    model = create_new_model()

    print("Streaming initial batch into model...")

    # Just a dictionary to keep track of clusters for each log
    cluster_results = {}

    # Step 5: Loop through each row (log entry) one by one
    for _, log in df.iterrows():


        embedding_vector = get_text_embedding(log["message"])

         #Create a single data record (dictionary) for this log
        x = {
            "level": log["level"],
            "source": log["source"],
        }

        # Add all 384 embedding values as "vec_0" to "vec_383"
        for i in range(len(embedding_vector)):
            feature_name = f"vec_{i}"
            x[feature_name] = embedding_vector[i]

        # Step 8: Learn scaling/encoding for this row and transform it
        pipeline.learn_one(x)
        processed_x = pipeline.transform_one(x)

        # Step 9: Feed it into the model
        model.learn_one(processed_x)

        # Optional: You could assign a cluster ID here
        # cluster_id = model.predict_one(processed_x)
        # cluster_results[log["log_id"]] = cluster_id

    # Step 10: Training complete
    print("Initial training complete.")
    print("Model now has", len(model.micro_clusters), "micro-clusters.")

    # Step 11: Save model and pipeline to disk
    save_model(model, pipeline)


if __name__ == "__main__":
    main()
