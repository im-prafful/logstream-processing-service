import sys
import pandas as pd

sys.path.append(sys.path[0] + "/..")

from src.db_connector import get_db_engine, fetch_logs_batch, save_embedding,save_pattern
from src.pipeline import get_text_embedding, build_feature_dict
from src.model import load_model, save_model


def main():
    print("Loading model and pipeline...")
    model, pipeline = load_model()

    if pipeline is None:
        print("ERROR: No pipeline found. Run runTrainingBatch.py first.")
        return

    #pipeline = pipeline.freeze()

    engine = get_db_engine()

    query = """
        SELECT *
        FROM logs
        WHERE log_id NOT IN (SELECT log_id FROM log_embeddings)
        AND level IN ('error','warning')
        ORDER BY log_id ASC
        LIMIT 2000;
    """

    df_new = fetch_logs_batch(engine, query)

    if df_new.empty:
        print("No new logs found. Everything is processed.")
        return

    print(f" Processing {len(df_new)} new logs...")

    for _, log in df_new.iterrows():
        log_id = log["log_id"]
        app_id = log["app_id"]
        level = log["level"]
        source = log["source"]

        # build full text
        full_text = f"{log['message']}. Parsed Context: {log['parsed_data']}"

        embedding_vector = get_text_embedding(full_text)
        feature_dict = build_feature_dict(level, source, embedding_vector)

        processed_features = pipeline.transform_one(feature_dict)
        model.learn_one(processed_features)

        cluster_id = model.predict_one(processed_features)

        save_embedding(
            engine=engine,
            log_id=log_id,
            app_id=app_id,
            embedding_vector=embedding_vector,
            cluster_id=cluster_id,
            level=level,
            source=source,
        )

    print("Saving updated model and pipeline...")
    print('--------------------')
    save_model(model, pipeline)

    save_pattern(engine=engine)
    print('new patterns saved to pattern table')
    print('------------------------')
    print("Incremental batch complete.")


if __name__ == "__main__":
    main()