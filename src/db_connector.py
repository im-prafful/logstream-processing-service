import pandas as pd
from sqlalchemy import create_engine

DB_USER = "masterUser"
DB_PASS = "Admin$1234"
DB_NAME = "Logstream_DB"
DB_HOST = "localhost"
DB_PORT = "5433"


def get_db_engine():
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        raise


def fetch_logs_batch(engine, query: str):
    print(f"Executing query: {query}")
    try:
        df = pd.read_sql(query, engine)
        print(f"Loaded {len(df)} logs.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
