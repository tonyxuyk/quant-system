import pandas as pd
import numpy as np
from typing import Dict, Any

class TradeExecutor:
    """交易器，支持多策略、止损止盈、仓位控制"""
    def __init__(self, commission=0.001, slippage=0.0, tax=0.001):
        self.commission = commission
        self.slippage = slippage
        self.tax = tax

    def dual_ma_strategy(self, df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
        ma_fast = df['Close'].rolling(fast).mean()
        ma_slow = df['Close'].rolling(slow).mean()
        signal = (ma_fast > ma_slow).astype(int)
        return signal.diff().fillna(0)

    def macd_strategy(self, df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        signal = (macd > macd_signal).astype(int)
        return signal.diff().fillna(0)

    def breakout_strategy(self, df: pd.DataFrame, window=20) -> pd.Series:
        high = df['High'].rolling(window).max()
        signal = (df['Close'] > high.shift(1)).astype(int)
        return signal.diff().fillna(0)

    def apply_stop(self, df: pd.DataFrame, signal: pd.Series, stop_loss=0.05, take_profit=0.1) -> pd.Series:
        position = 0
        entry_price = 0
        result = []
        for i, row in df.iterrows():
            if signal.loc[i] == 1:
                position = 1
                entry_price = row['Close']
            elif signal.loc[i] == -1:
                position = 0
            if position == 1:
                if row['Close'] <= entry_price * (1 - stop_loss):
                    position = 0
                elif row['Close'] >= entry_price * (1 + take_profit):
                    position = 0
            result.append(position)
        return pd.Series(result, index=df.index)

    def position_control(self, capital: float, price: float, risk_pct: float = 0.1) -> int:
        return int((capital * risk_pct) // price) 