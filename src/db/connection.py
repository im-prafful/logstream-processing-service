from sqlalchemy import create_engine

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
