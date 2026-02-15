import pandas as pd
from sqlalchemy import text


def fetch_logs_batch(engine, query: str):
    """Fetch dataframe from DB using a SELECT query."""
    print(f"Executing query:\n{query}")
    try:
        df = pd.read_sql(query, engine)
        print(f"Loaded {len(df)} logs.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def fetch_min_timestamp(engine, timestamp_query):
    print(f"Fetching timestamp of the latest unprocessed log")
    try:
        with engine.begin() as conn:
            result = conn.execute(timestamp_query)

            # Fetch the first row
            row = result.fetchone()

            if row is None:
                print("No timestamp found")
                return None

            # Extract the timestamp (first column)
            timestamp_value = row[0]

            print(
                f"Fetched timestamp: {timestamp_value}, Type: {type(timestamp_value)}"
            )

            return timestamp_value  # Returns datetime.datetime object

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def save_embedding(engine, log_id, app_id, embedding_vector, cluster_id, level, source):
    """Insert embedding + cluster into log_embeddings table."""

    insert_query = text(
        """
        INSERT INTO log_embeddings (
            log_id, app_id, embedding, cluster_id, level, source
        )
        VALUES (:log_id, :app_id, :embedding, :cluster_id, :level, :source)
        ON CONFLICT (log_id) DO NOTHING;
    """
    )

    # Insert  cluster_id into logs table."""
    insert_to_logs = text(
        """
            UPDATE logs 
            SET cluster_id = :cluster_id
            WHERE log_id = :log_id;
        """
    )

    with engine.begin() as conn:
        # Step A: Execute INSERT/UPDATE on log_embeddings
        conn.execute(
            insert_query,
            {
                "log_id": log_id,
                "app_id": app_id,
                "embedding": embedding_vector.tolist(),
                "cluster_id": cluster_id,
                "level": level,
                "source": source,
            },
        )

        # Step B: Execute UPDATE on logs table in the SAME TRANSACTION
        conn.execute(insert_to_logs, {"log_id": log_id, "cluster_id": cluster_id})
