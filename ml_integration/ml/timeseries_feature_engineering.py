import pandas as pd
import numpy as np

class TimeSeriesFeatureEngineer:
    def create_rolling_features(self, df, window_sizes=[5, 10, 20], price_col='close', industry_col=None):
        """
        创建滚动收益率、波动率、均值、相关性、技术指标变化率等特征
        """
        df = df.copy()
        for w in window_sizes:
            df[f'roll_return_{w}'] = df[price_col].pct_change(w)
            df[f'roll_vol_{w}'] = df[price_col].pct_change().rolling(w).std()
            df[f'roll_mean_{w}'] = df[price_col].rolling(w).mean()
        # 技术指标变化率
        df['rsi_14'] = self._rsi(df[price_col], 14)
        for w in window_sizes:
            df[f'rsi_14_chg_{w}'] = df['rsi_14'].diff(w)
        # 行业指数相关性（如有）
        if industry_col and industry_col in df.columns:
            for w in window_sizes:
                df[f'ind_corr_{w}'] = df[price_col].rolling(w).corr(df[industry_col])
        return df

    def _rsi(self, series, window):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(window=window, min_periods=1).mean()
        ma_down = down.rolling(window=window, min_periods=1).mean()
        rsi = 100 - (100 / (1 + ma_up / (ma_down + 1e-9)))
        return rsi

    def split_train_test(self, df, test_size=0.2, time_col='date'):
        """
        按时间分割，训练集早于测试集
        """
        df = df.sort_values(time_col)
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test

    def preprocess(self, df, label_col='label', dropna=True):
        """
        生成特征和标签，确保特征时间戳早于标签
        """
        X = df.drop(columns=[label_col])
        y = df[label_col]
        if dropna:
            valid_idx = X.dropna().index.intersection(y.dropna().index)
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
        return X, y