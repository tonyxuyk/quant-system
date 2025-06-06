#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from strategy_core.strategy_core import MovingAverageCrossStrategy
from strategy_core.backtest_engine import BacktestEngine
import pandas as pd
import numpy as np

def test_strategy_and_backtest():
    """测试策略和回测引擎"""
    print("🧪 测试策略和回测引擎...")
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.normal(0, 1, 100)),
        'open': 100 + np.cumsum(np.random.normal(0, 1, 100)),
        'high': 100 + np.cumsum(np.random.normal(0, 1, 100)),
        'low': 100 + np.cumsum(np.random.normal(0, 1, 100)),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    print(f"📊 测试数据: {len(data)} 行")
    print(f"价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 测试策略
    strategy = MovingAverageCrossStrategy(fast=5, slow=20)
    signals = strategy.generate_signal(data)
    print(f"📈 信号数量: {signals.sum()}")
    print(f"买入信号: {(signals == 1).sum()}")
    print(f"卖出信号: {(signals == -1).sum()}")
    
    # 测试回测
    engine = BacktestEngine(commission=0.001, tax=0.001)
    
    def strategy_func(df):
        return strategy.generate_signal(df)
    
    results = engine.run(
        stock_data={'TEST': data},
        strategy_func=strategy_func,
        initial_cash=100000
    )
    
    print(f"💰 总收益: {results.get('total_return', 0):.4f}")
    print(f"📊 年化收益: {results.get('annual_return', 0):.4f}")
    print(f"📉 最大回撤: {results.get('max_drawdown', 0):.4f}")
    print(f"🎯 夏普比率: {results.get('sharpe', 0):.4f}")
    print(f"🔄 交易次数: {len(results.get('trades', []))}")
    
    if len(results.get('trades', [])) > 0:
        trades_df = results['trades']
        print(f"📋 交易记录样本:")
        print(trades_df.head())
    
    return results

if __name__ == "__main__":
    test_strategy_and_backtest() 