import sys
sys.path.append(".")
from src.db.connection import get_db_engine
import pandas as pd
engine = get_db_engine()
try:
    print(pd.read_sql("SELECT enumlabel FROM pg_enum WHERE enumtypid = 'incident_status'::regtype;", engine))
except Exception as e:
    print(e)
