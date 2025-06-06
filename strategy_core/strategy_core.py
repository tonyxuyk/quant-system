from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("警告: PyYAML未安装，YAML配置功能将不可用")

@dataclass
class MarketData:
    open: float
    high: float
    low: float
    close: float
    volume: float

class TradingStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data: MarketData) -> str:
        pass
    @abstractmethod
    def get_parameters(self) -> dict:
        pass

# --- 价格行为学策略 ---
class HammerStrategy(TradingStrategy):
    def __init__(self, min_wick_ratio: float = 0.7):
        self.min_wick_ratio = min_wick_ratio
    def is_hammer(self, candle: MarketData) -> bool:
        body_size = abs(candle.close - candle.open)
        lower_wick = min(candle.open, candle.close) - candle.low
        upper_wick = candle.high - max(candle.open, candle.close)
        return (lower_wick > body_size * self.min_wick_ratio and upper_wick < body_size * 0.3)
    def generate_signal(self, data: MarketData) -> str:
        if self.is_hammer(data):
            return "BUY"
        return "HOLD"
    def get_parameters(self):
        return {'min_wick_ratio': self.min_wick_ratio}

class EngulfingStrategy(TradingStrategy):
    def __init__(self, min_body_ratio: float = 1.0):
        self.min_body_ratio = min_body_ratio
    def is_bullish_engulfing(self, prev: MarketData, curr: MarketData) -> bool:
        prev_body = prev.close - prev.open
        curr_body = curr.close - curr.open
        return (prev_body < 0 and curr_body > 0 and abs(curr_body) > abs(prev_body) * self.min_body_ratio and curr.open < prev.close and curr.close > prev.open)
    def generate_signal(self, data: List[MarketData]) -> str:
        if len(data) < 2:
            return "HOLD"
        if self.is_bullish_engulfing(data[-2], data[-1]):
            return "BUY"
        return "HOLD"
    def get_parameters(self):
        return {'min_body_ratio': self.min_body_ratio}

class HeadAndShouldersStrategy(TradingStrategy):
    def __init__(self, min_ratio: float = 0.95):
        self.min_ratio = min_ratio
    def is_head_and_shoulders(self, candles: List[MarketData]) -> bool:
        if len(candles) < 7:
            return False
        highs = [c.high for c in candles[-7:]]
        mid = highs[3]
        left = max(highs[0:3])
        right = max(highs[4:7])
        return mid > left * self.min_ratio and mid > right * self.min_ratio and left > right * 0.9
    def generate_signal(self, data: List[MarketData]) -> str:
        if self.is_head_and_shoulders(data):
            return "SELL"
        return "HOLD"
    def get_parameters(self):
        return {'min_ratio': self.min_ratio}

# --- 示例价格行为策略（锤子线）---
class PriceActionStrategy(TradingStrategy):
    def __init__(self, min_wick_ratio: float = 0.7):
        self.min_wick_ratio = min_wick_ratio
    def is_hammer(self, candle: MarketData) -> bool:
        body_size = abs(candle.close - candle.open)
        lower_wick = min(candle.open, candle.close) - candle.low
        return (lower_wick > body_size * self.min_wick_ratio and (candle.high - max(candle.open, candle.close)) < body_size * 0.3)
    def generate_signal(self, data: MarketData) -> str:
        if self.is_hammer(data):
            return "BUY"
        return "HOLD"
    def get_parameters(self):
        return {'min_wick_ratio': self.min_wick_ratio}

# --- 执行计划与触发器 ---
class Trigger:
    TIME_BASED = "TIME_BASED"
    EVENT_BASED = "EVENT_BASED"
    CONDITION_BASED = "CONDITION_BASED"
    def __init__(self, trigger_type: str, value: Any):
        self.trigger_type = trigger_type
        self.value = value
    def check(self, data: MarketData) -> bool:
        if self.trigger_type == self.TIME_BASED:
            # 这里仅做接口，实际需集成调度器
            return True
        elif self.trigger_type == self.EVENT_BASED:
            # 事件触发（如gap_up等）
            return data.open > data.close * 1.05 if self.value == 'gap_up' else False
        elif self.trigger_type == self.CONDITION_BASED:
            return self.value(data)
        return False

class ExecutionPlan:
    def __init__(self, strategy: TradingStrategy, trigger: Trigger):
        self.strategy = strategy
        self.trigger = trigger
    def set_strategy(self, strategy: TradingStrategy):
        self.strategy = strategy
    def execute(self, data: Any) -> str:
        if self.trigger.check(data):
            return self.strategy.generate_signal(data)
        return "NO_ACTION"

# --- 趋势跟踪策略 ---
class MovingAverageCrossStrategy(TradingStrategy):
    def __init__(self, fast: int = 5, slow: int = 20):
        self.fast = fast
        self.slow = slow
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        ma_fast = data['close'].rolling(self.fast).mean()
        ma_slow = data['close'].rolling(self.slow).mean()
        signal = (ma_fast > ma_slow).astype(int)
        return signal.diff().fillna(0)  # 返回1, -1, 0
    def get_parameters(self):
        return {'fast': self.fast, 'slow': self.slow}

class MACDStrategy(TradingStrategy):
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal, adjust=False).mean()
        signal = (macd > macd_signal).astype(int)
        return signal.diff().fillna(0)  # 返回1, -1, 0
    def get_parameters(self):
        return {'fast': self.fast, 'slow': self.slow, 'signal': self.signal}

class MomentumBreakoutStrategy(TradingStrategy):
    def __init__(self, window: int = 20):
        self.window = window
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].rolling(self.window).max()
        signal = (data['close'] > high.shift(1)).astype(int)
        return signal.diff().fillna(0)  # 返回1, -1, 0
    def get_parameters(self):
        return {'window': self.window}

# --- 均值回归策略 ---
class RSIStrategy(TradingStrategy):
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        signal = pd.Series(0, index=data.index)  # 默认为0(HOLD)
        signal[rsi > self.overbought] = -1  # SELL
        signal[rsi < self.oversold] = 1     # BUY
        return signal
    def get_parameters(self):
        return {'period': self.period, 'overbought': self.overbought, 'oversold': self.oversold}

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, window: int = 20, num_std: float = 2):
        self.window = window
        self.num_std = num_std
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        ma = data['close'].rolling(self.window).mean()
        std = data['close'].rolling(self.window).std()
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        signal = pd.Series(0, index=data.index)  # 默认为0(HOLD)
        signal[data['close'] > upper] = -1  # SELL
        signal[data['close'] < lower] = 1   # BUY
        return signal
    def get_parameters(self):
        return {'window': self.window, 'num_std': self.num_std}

# --- 机器学习策略（接口/示例） ---
class SklearnMLStrategy(TradingStrategy):
    def __init__(self, model, feature_cols: List[str], threshold: float = 0.5):
        self.model = model
        self.feature_cols = feature_cols
        self.threshold = threshold
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        proba = self.model.predict_proba(data[self.feature_cols])[:, 1]
        signal = np.where(proba > self.threshold, 'BUY', 'HOLD')
        return pd.Series(signal, index=data.index)
    def get_parameters(self):
        return {'model': str(self.model), 'feature_cols': self.feature_cols, 'threshold': self.threshold}

class LSTMStrategy(TradingStrategy):
    def __init__(self, model, feature_cols: List[str], threshold: float = 0.5):
        self.model = model
        self.feature_cols = feature_cols
        self.threshold = threshold
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 伪代码，实际需用torch/keras模型
        proba = self.model.predict(data[self.feature_cols])
        signal = np.where(proba > self.threshold, 'BUY', 'HOLD')
        return pd.Series(signal, index=data.index)
    def get_parameters(self):
        return {'model': str(self.model), 'feature_cols': self.feature_cols, 'threshold': self.threshold}

# --- 策略工厂支持YAML配置 ---
class StrategyFactory:
    @classmethod
    def create_from_config(cls, config: dict) -> TradingStrategy:
        type_map = {
            'Hammer': HammerStrategy,
            'Engulfing': EngulfingStrategy,
            'HeadAndShoulders': HeadAndShouldersStrategy,
            'MACD': MACDStrategy,
            'MA_Cross': MovingAverageCrossStrategy,
            'MomentumBreakout': MomentumBreakoutStrategy,
            'RSI': RSIStrategy,
            'Bollinger': BollingerBandsStrategy,
            'SklearnML': SklearnMLStrategy,
            'LSTM': LSTMStrategy,
            'PriceAction': PriceActionStrategy
        }
        stype = config['type']
        params = config.get('params', {})
        if stype in type_map:
            return type_map[stype](**params)
        raise ValueError(f'未知策略类型: {stype}')

# --- 策略组合回测 ---
class StrategyPortfolio:
    def __init__(self, strategies: List[TradingStrategy]):
        self.strategies = strategies
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        for i, strat in enumerate(self.strategies):
            sig = strat.generate_signal(data)
            signals[f'strategy_{i}'] = sig
        # 简单合成：若有BUY则BUY，若有SELL则SELL，否则HOLD
        signals['final'] = signals.apply(lambda row: 'BUY' if 'BUY' in row.values else ('SELL' if 'SELL' in row.values else 'HOLD'), axis=1)
        return signals

# --- YAML配置解析与组合策略示例 ---
def load_strategies_from_yaml(yaml_path: str) -> List[TradingStrategy]:
    if not HAS_YAML:
        raise ImportError("PyYAML未安装，无法加载YAML配置文件")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    strategies = []
    for sconf in config['strategies']:
        strategies.append(StrategyFactory.create_from_config(sconf))
    return strategies

# 示例YAML:
# strategies:
#   - name: "趋势MACD"
#     type: "MACD"
#     params:
#       fast: 12
#       slow: 26
#       signal: 9
#   - name: "均值RSI"
#     type: "RSI"
#     params:
#       period: 14
#       overbought: 70
#       oversold: 30
#
# portfolio = StrategyPortfolio(load_strategies_from_yaml('strategies.yaml'))
# signals = portfolio.generate_signals(data)
# print(signals.head())

# --- 动态切换演示 ---
if __name__ == "__main__":
    # 示例数据
    candle = MarketData(open=10, high=12, low=9, close=11, volume=10000)
    hammer = HammerStrategy(min_wick_ratio=0.7)
    engulfing = EngulfingStrategy(min_body_ratio=1.0)
    # 触发器：定时
    trigger = Trigger(Trigger.TIME_BASED, "09:30")
    plan = ExecutionPlan(strategy=hammer, trigger=trigger)
    print("初始策略:", plan.execute(candle))
    # 动态切换策略
    plan.set_strategy(PriceActionStrategy(min_wick_ratio=0.8))
    print("切换后策略:", plan.execute(candle)) 