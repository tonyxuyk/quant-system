import pandas as pd
import sqlite3
import os
from datetime import datetime

class DataCache:
    """数据缓存工具，支持CSV和SQLite"""
    def __init__(self, data_dir='data', db_path='data/market_data.db'):
        self.data_dir = data_dir
        self.db_path = db_path
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.close()

    def save_csv(self, df, symbol, freq):
        fname = f"{self.data_dir}/{symbol}_{freq}.csv"
        df.to_csv(fname)
        return fname

    def load_csv(self, symbol, freq):
        fname = f"{self.data_dir}/{symbol}_{freq}.csv"
        if os.path.exists(fname):
            return pd.read_csv(fname, index_col=0, parse_dates=True)
        return None

    def save_sqlite(self, df, symbol, freq):
        conn = sqlite3.connect(self.db_path)
        table = f"{symbol}_{freq}"
        df.to_sql(table, conn, if_exists='replace')
        conn.close()

    def load_sqlite(self, symbol, freq):
        conn = sqlite3.connect(self.db_path)
        table = f"{symbol}_{freq}"
        try:
            df = pd.read_sql(f"SELECT * FROM '{table}'", conn, index_col='index', parse_dates=['index'])
        except Exception:
            df = None
        conn.close()
        return df

    def incremental_update_csv(self, df_new, symbol, freq):
        df_old = self.load_csv(symbol, freq)
        if df_old is not None:
            df = pd.concat([df_old, df_new]).drop_duplicates().sort_index()
        else:
            df = df_new
        self.save_csv(df, symbol, freq)
        return df

    def incremental_update_sqlite(self, df_new, symbol, freq):
        df_old = self.load_sqlite(symbol, freq)
        if df_old is not None:
            df = pd.concat([df_old, df_new]).drop_duplicates().sort_index()
        else:
            df = df_new
        self.save_sqlite(df, symbol, freq)
        return df 