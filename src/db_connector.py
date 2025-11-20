import pandas as pd
from sqlalchemy import create_engine, text

DB_USER = "masterUser"
DB_PASS = "Admin$1234"
DB_NAME = "Logstream_DB"
DB_HOST = "localhost"
DB_PORT = "5433"


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

    #Insert  cluster_id into logs table."""
    insert_to_logs=text(
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
        conn.execute(
            insert_to_logs,
            {
                "log_id": log_id,
                "cluster_id": cluster_id
            }
        )

def save_pattern(engine):

    ## Fetch the first log entry for each cluster by joining logs with a subquery
    # that selects MIN(log_id) (the earliest entry) per cluster_id.
    # Also returns the total number of logs in each cluster.
    # The result gives one representative log per cluster (the "pattern")
    # along with metadata needed to store log patterns later.

    query = text("""
        SELECT 
            concat_ws(' | ', l.source, l.level, l.message, l.parsed_data) AS merged_string,
            l.cluster_id,
            t.total_count
        FROM logs l
        JOIN (
            SELECT cluster_id, MIN(log_id) AS first_log, COUNT(*) AS total_count
            FROM logs
            GROUP BY cluster_id
        ) t
        ON l.cluster_id = t.cluster_id AND l.log_id = t.first_log;
    """)

    with engine.begin() as conn:
        result = conn.execute(query)
        rows = result.fetchall()   



    insert_pattern_query = text(
        """
        INSERT INTO log_patterns (
            app_id, log_template, incident_count, cluster_id
        )
        -- SELECT the necessary columns from the logs table (aliased as l)
        SELECT 
            l.app_id,                     -- Fetch the app_id
            l.message AS log_template,    -- Use the 'message' column as the 'log_template'
            1 AS incident_count,          -- Initial count for this pattern
            :cluster_id AS cluster_id     -- Use the cluster_id passed to the function
        FROM logs l
        WHERE l.log_id = :log_id
        -- Prevent duplicates if called again with the same log_id
        ON CONFLICT (pattern_id) DO NOTHING;
        """
    )






    