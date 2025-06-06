import pandas as pd
import numpy as np

class FeatureEngineer:
    """特征工程工具"""
    @staticmethod
    def add_rsi(df, window=14, col='Close'):
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9, col='Close'):
        ema_fast = df[col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[col].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        df['MACD'] = macd
        df['MACD_signal'] = signal_line
        df['MACD_hist'] = macd - signal_line
        return df

    @staticmethod
    def add_bollinger(df, window=20, col='Close'):
        ma = df[col].rolling(window).mean()
        std = df[col].rolling(window).std()
        df['BOLL_MID'] = ma
        df['BOLL_UP'] = ma + 2 * std
        df['BOLL_LOW'] = ma - 2 * std
        return df

    @staticmethod
    def add_fundamental_features(df, financials):
        """合并基本面特征（如市盈率/ROE/现金流）"""
        for key, fin_df in financials.items():
            if fin_df is not None and not fin_df.empty:
                df = df.merge(fin_df, how='left', left_index=True, right_on=fin_df.columns[1])
        return df

    @staticmethod
    def add_momentum(df, window=10, col='Close'):
        """动量指标（MOM）"""
        df['MOM'] = df[col] - df[col].shift(window)
        return df

    @staticmethod
    def add_stochastic_oscillator(df, k_window=14, d_window=3, col='Close'):
        """随机震荡指标（Stochastic Oscillator, KDJ）"""
        low_min = df[col].rolling(window=k_window).min()
        high_max = df[col].rolling(window=k_window).max()
        df['%K'] = 100 * (df[col] - low_min) / (high_max - low_min)
        df['%D'] = df['%K'].rolling(window=d_window).mean()
        return df

    @staticmethod
    def add_obv(df, price_col='Close', vol_col='Volume'):
        """成交量平衡指标（OBV）"""
        obv = [0]
        for i in range(1, len(df)):
            if df[price_col].iloc[i] > df[price_col].iloc[i-1]:
                obv.append(obv[-1] + df[vol_col].iloc[i])
            elif df[price_col].iloc[i] < df[price_col].iloc[i-1]:
                obv.append(obv[-1] - df[vol_col].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        return df

    @staticmethod
    def add_bias(df, window=6, col='Close'):
        """市场情绪指标：乖离率（BIAS）"""
        ma = df[col].rolling(window=window).mean()
        df['BIAS'] = (df[col] - ma) / ma * 100
        return df 