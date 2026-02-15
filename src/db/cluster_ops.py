import pandas as pd
from sqlalchemy import text


def save_cluster_stats(engine, batch_stats: dict):
    """
    Saves current batch stats to history.
    batch_stats format: {cluster_id: count, ...}
    """
    if not batch_stats:
        return

    # We timestamp this entry as 'NOW()' so we know when this batch happened
    insert_query = text(
        """
        INSERT INTO cluster_volume_history (cluster_id, log_count, batch_timestamp)
        VALUES (:cluster_id, :log_count, NOW())
    """
    )

    params = [
        {"cluster_id": cid, "log_count": count} for cid, count in batch_stats.items()
    ]

    try:
        with engine.begin() as conn:
            conn.execute(insert_query, params)
            print(f"Saved volume stats for {len(params)} clusters.")
    except Exception as e:
        print(f"Error saving cluster stats: {e}")


def fetch_cluster_history(engine, window_size=5):
    """
    Fetches the last N counts for ALL clusters to build the context window.
    Returns: DataFrame with columns [cluster_id, log_count, batch_timestamp]
    """
    # This query retrieves the most recent 'window_size' entries for every cluster
    query = text(
        f"""
        WITH ranked_history AS (
            SELECT 
                cluster_id, 
                log_count, 
                batch_timestamp,
                ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY batch_timestamp DESC) as rn
            FROM cluster_volume_history
        )
        SELECT cluster_id, log_count, batch_timestamp
        FROM ranked_history
        WHERE rn <= :window_size
        ORDER BY cluster_id, batch_timestamp ASC;
    """
    )

    try:
        df = pd.read_sql(query, engine, params={"window_size": window_size})
        return df
    except Exception as e:
        print(f"Error fetching history: {e}")
        return pd.DataFrame()
