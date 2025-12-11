import pandas as pd
from sqlalchemy import create_engine, text

DB_USER = "masterUser"
DB_PASS = "Admin$1234"
DB_NAME = "LogStream_2.0"
DB_HOST = "localhost"
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
