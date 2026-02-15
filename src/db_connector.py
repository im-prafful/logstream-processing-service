import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import math

DB_USER = "masterUser"
DB_PASS = "Admin$1234"
DB_NAME = "LogStream_2.0"
DB_HOST = "logstream-2-db.czegikcsabng.ap-south-1.rds.amazonaws.com"
DB_PORT = "5432"


def get_db_engine():
    """Create SQLAlchemy engine."""
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        raise


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


def save_pattern(engine):
    """
    Fetches one representative log (pattern) for each cluster
    and inserts it into the log_patterns table.
    """

    # Fetch the first log entry for each cluster (the pattern)
    # along with the cluster's total count.
    query = text(
        """
        SELECT 
            concat_ws(' | ', l.source, l.level, l.message, l.parsed_data) AS merged_string,
            l.cluster_id,
            l.app_id, -- Need app_id for the insert
            t.total_count
        FROM logs l
        JOIN (
            SELECT cluster_id, MIN(log_id) AS first_log, COUNT(*) AS total_count
            FROM logs
            GROUP BY cluster_id
            HAVING cluster_id IS NOT NULL -- Only consider logs that have been clustered
        ) t
        ON l.cluster_id = t.cluster_id AND l.log_id = t.first_log;
    """
    )

    # Insert log patterns into the log_patterns table.
    insert_pattern_query = text(
        """
        INSERT INTO log_patterns (
            app_id, log_template, incident_count, cluster_id
        )
        VALUES (
            :app_id, 
            :log_template, 
            :incident_count, 
            :cluster_id
        )
        """
    )

    with engine.begin() as conn:
        # Step A: Fetch all pattern data
        result = conn.execute(query)
        # Fetch all rows, the schema is (merged_string, cluster_id, app_id, total_count)
        rows = result.fetchall()

        print(f"Fetched {len(rows)} distinct log patterns.")

        # Step B: Prepare parameters for bulk insertion
        insert_params = [
            {
                "app_id": row[2],  # app_id
                "log_template": row[0],  # merged_string
                "incident_count": row[
                    3
                ],  # total_count (or 1 depending on intent, using total_count here)
                "cluster_id": row[1],  # cluster_id
            }
            for row in rows
        ]

        # Step C: Execute  insertion
        if insert_params:
            conn.execute(insert_pattern_query, insert_params)
            print(f"Inserted/updated {len(insert_params)} log patterns.")


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


def create_incident(engine, cluster_id, reason="Volume Anomaly"):
    """
    Simple function to trigger an incident. Logic for 'detecting' it is now elsewhere.
    """
    check_query = text(
        "SELECT 1 FROM incidents WHERE cluster_id = :cid AND state = 'OPEN'"
    )

    update_query = text(
        """
        UPDATE incidents SET last_seen_at = NOW() 
        WHERE cluster_id = :cid AND state = 'OPEN'
    """
    )

    insert_query = text(
        """
        INSERT INTO incidents (cluster_id, state, last_seen_at) 
        VALUES (:cid, 'OPEN', NOW())
    """
    )

    with engine.begin() as conn:
        exists = conn.execute(check_query, {"cid": cluster_id}).fetchone()

        if exists:
            conn.execute(update_query, {"cid": cluster_id})
            print(f"⚠️ Incident UPDATED for Cluster {cluster_id}")
        else:
            conn.execute(insert_query, {"cid": cluster_id})
            print(f"🚨 New Incident CREATED for Cluster {cluster_id} [{reason}]")


def detect_and_create_incidents(engine, batch_size, global_timestamp):
    """
    End-of-batch orchestrator: saves cluster volume stats,
    runs anomaly detection, and creates incidents for flagged clusters.
    """
    from src.volume_analyzer import VolumeAnomalyDetector

    # 1. Count how many logs landed in each cluster during this batch
    count_query = text("""
        SELECT cluster_id, COUNT(*) as cnt
        FROM logs
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
    """)

    try:
        with engine.begin() as conn:
            rows = conn.execute(count_query).fetchall()
        batch_stats = {row[0]: row[1] for row in rows}
    except Exception as e:
        print(f"Error counting cluster stats: {e}")
        return

    # 2. Save stats to history
    save_cluster_stats(engine, batch_stats)

    # 3. Fetch history window
    history_df = fetch_cluster_history(engine, window_size=5)

    # 4. Load volume model and detect anomalies
    vol_detector = VolumeAnomalyDetector(window_size=5)
    vol_detector.load("models/production")
    anomalous_clusters = vol_detector.detect_anomalies(history_df)

    # 5. Create incidents
    if anomalous_clusters:
        print(f"🚨 Detected {len(anomalous_clusters)} anomalous clusters!")
        for cid in anomalous_clusters:
            create_incident(engine, cid, reason="Volume Anomaly")
    else:
        print("✅ No volume anomalies detected.")
