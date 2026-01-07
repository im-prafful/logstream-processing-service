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


def calculate_dynamic_threshold(
    total_batch_size: int,
    cluster_incident_count: int,
    total_clusters: int,
    sensitivity: float = 1.0,
) -> int:
    """
    Calculate dynamic threshold based on current batch behavior.

    Parameters:
    - total_batch_size: Total logs in current batch
    - cluster_incident_count: Logs in this specific cluster in current batch
    - total_clusters: Total number of clusters
    - sensitivity: Tuning parameter (higher = harder to trigger)

    Returns:
    - threshold: Minimum logs needed to trigger incident
    """

    # Average logs per cluster in this batch
    average_per_cluster = total_batch_size / total_clusters if total_clusters > 0 else 0

    # If a cluster has 2x the average, it's potentially anomalous
    # Divide by sensitivity so higher sensitivity = lower threshold (easier to trigger)
    threshold = (average_per_cluster * 2.0) / sensitivity

    # Bounds: minimum 10, maximum 80% of what this cluster actually has
    threshold = max(10, min(threshold, int(cluster_incident_count * 0.8)))

    return int(threshold)


def detect_and_create_incidents(engine, batch_size: int, global_timestamp):
    """
    Detect abnormal clusters using dynamic threshold.

    Parameters:
    - engine: SQLAlchemy engine
    - batch_size: Total number of logs processed in this incremental batch
    """

    with engine.begin() as conn:
        # Get total number of clusters from the pattern table
        total_clusters_result = conn.execute(
            text(
                """
            SELECT COUNT(DISTINCT cluster_id) as total_clusters
            FROM log_patterns
        """
            )
        ).fetchone()

        total_clusters = total_clusters_result[0]

        # Get log count per cluster for recent logs
        cluster_result = conn.execute(
            text(
                """
            SELECT cluster_id, COUNT(*) as log_count
            FROM logs 
            WHERE timestamp >= :global_timestamp
              AND level IN ('error', 'warning')
              AND cluster_id IS NOT NULL
            GROUP BY cluster_id
        """
            ),
            {"global_timestamp": global_timestamp},
        )  # ←PARAMETER BINDING

        rows = cluster_result.fetchall()

        print(f"Analyzing {len(rows)} clusters for anomalies...")

        for row in rows:
            cluster_id = row[0]
            cluster_log_count = row[1]

            # Calculate dynamic threshold
            threshold = calculate_dynamic_threshold(
                total_batch_size=batch_size,
                cluster_incident_count=cluster_log_count,
                total_clusters=total_clusters,
                sensitivity=1.0,
            )

            print(
                f"Cluster {cluster_id}: log_count={cluster_log_count}, threshold={threshold}"
            )

            # Only create incident if log_count exceeds dynamic threshold
            if cluster_log_count >= threshold:
                # Check if incident already exists
                exists = conn.execute(
                    text(
                        """
                        SELECT 1
                        FROM incidents
                        WHERE cluster_id = :cluster_id
                          AND state = 'OPEN'
                    """
                    ),
                    {"cluster_id": cluster_id},
                ).fetchone()

                if exists:
                    conn.execute(
                        text(
                            """
                            UPDATE incidents
                            SET last_seen_at = now()
                            WHERE cluster_id = :cluster_id
                              AND state = 'OPEN'
                        """
                        ),
                        {"cluster_id": cluster_id},
                    )
                    print(f"Updated incident for cluster {cluster_id}")
                else:
                    conn.execute(
                        text(
                            """
                            INSERT INTO incidents (
                                cluster_id, state, last_seen_at
                            )
                            VALUES (
                                :cluster_id, 'OPEN', now()
                            )
                        """
                        ),
                        {"cluster_id": cluster_id},
                    )
                    print(
                        f"INCIDENT CREATED for cluster {cluster_id} (log_count: {cluster_log_count}, threshold: {threshold})"
                    )
