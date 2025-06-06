#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tony&Associates QuantAI Trader - 集成测试
测试系统各模块的集成功能
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from integration import (
        QuantSystem, DataEngine, StrategyEngine, MLEngine,
        create_quant_system, quick_backtest
    )
except ImportError as e:
    print(f"导入失败: {e}")
    raise

class TestDataEngine(unittest.TestCase):
    """数据引擎测试"""
    
    def setUp(self):
        self.data_engine = DataEngine()
    
    def test_data_pipeline(self):
        """测试数据处理流水线"""
        start_date = "2024-01-01"
        end_date = "2024-02-01"
        symbol = "AAPL"
        
        result = self.data_engine.get_data_pipeline(symbol, start_date, end_date)
        
        # 验证结果结构
        self.assertEqual(result['status'], 'success')
        self.assertIn('processed_data', result)
        self.assertIn('features', result)
        
        # 验证数据类型
        self.assertIsInstance(result['processed_data'], pd.DataFrame)
        self.assertIsInstance(result['features'], pd.DataFrame)
        
        # 验证数据列
        data = result['processed_data']
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, data.columns)

class TestStrategyEngine(unittest.TestCase):
    """策略引擎测试"""
    
    def setUp(self):
        self.data_engine = DataEngine()
        self.strategy_engine = StrategyEngine(self.data_engine)
    
    def test_strategy_pipeline(self):
        """测试策略回测流水线"""
        start_date = "2024-01-01"
        end_date = "2024-02-01"
        symbol = "AAPL"
        strategy = "moving_average"
        
        result = self.strategy_engine.run_strategy_pipeline(
            symbol, strategy, start_date, end_date, 1000000
        )
        
        # 验证结果结构
        self.assertEqual(result['status'], 'success')
        self.assertIn('backtest_results', result)
        self.assertIn('signals', result)
        
        # 验证回测指标
        backtest = result['backtest_results']
        required_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 
            'volatility', 'win_rate'
        ]
        for metric in required_metrics:
            self.assertIn(metric, backtest)
            self.assertIsInstance(backtest[metric], (int, float))
    
    def test_signal_generation(self):
        """测试交易信号生成"""
        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # 测试双均线策略
        signals = self.strategy_engine._generate_signals("moving_average", test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

class TestMLEngine(unittest.TestCase):
    """机器学习引擎测试"""
    
    def setUp(self):
        self.data_engine = DataEngine()
        self.ml_engine = MLEngine(self.data_engine)
    
    def test_ml_pipeline(self):
        """测试机器学习流水线"""
        start_date = "2024-01-01"
        end_date = "2024-02-01"
        symbol = "AAPL"
        model_type = "xgboost"
        
        result = self.ml_engine.run_ml_pipeline(
            symbol, start_date, end_date, model_type
        )
        
        # 验证结果结构
        self.assertEqual(result['status'], 'success')
        self.assertIn('model_performance', result)
        self.assertIn('predictions', result)
        self.assertIn('feature_importance', result)
        
        # 验证性能指标
        performance = result['model_performance']
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            self.assertIn(metric, performance)
            self.assertIsInstance(performance[metric], (int, float))
            self.assertGreaterEqual(performance[metric], 0)
            self.assertLessEqual(performance[metric], 1)

class TestQuantSystem(unittest.TestCase):
    """量化系统集成测试"""
    
    def setUp(self):
        self.system = QuantSystem()
    
    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertIsNotNone(self.system.data_engine)
        self.assertIsNotNone(self.system.strategy_engine)
        self.assertIsNotNone(self.system.ml_engine)
    
    def test_full_pipeline(self):
        """测试完整流水线"""
        start_date = "2024-01-01"
        end_date = "2024-02-01"
        symbol = "AAPL"
        strategy = "moving_average"
        
        result = self.system.run_full_pipeline(
            symbol=symbol,
            strategy_name=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000,
            use_ml=True,
            optimize_params=False
        )
        
        # 验证策略结果
        self.assertIn('strategy', result)
        strategy_result = result['strategy']
        self.assertEqual(strategy_result['status'], 'success')
        
        # 验证ML结果（如果启用）
        if 'ml' in result:
            ml_result = result['ml']
            self.assertEqual(ml_result['status'], 'success')
        
        # 验证分析结果
        if 'analysis' in result:
            analysis = result['analysis']
            self.assertIn('recommendations', analysis)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效日期范围
        result = self.system.run_full_pipeline(
            symbol="INVALID",
            strategy_name="invalid_strategy",
            start_date="2025-01-01",  # 未来日期
            end_date="2025-02-01",
            initial_capital=1000000
        )
        
        # 应该能够处理错误但不崩溃
        self.assertIsInstance(result, dict)

class TestQuickFunctions(unittest.TestCase):
    """快速函数测试"""
    
    def test_create_quant_system(self):
        """测试系统创建函数"""
        system = create_quant_system()
        self.assertIsNotNone(system)
        self.assertIsInstance(system, QuantSystem)
    
    def test_quick_backtest(self):
        """测试快速回测函数"""
        result = quick_backtest("AAPL", "moving_average", 30)
        
        self.assertIsInstance(result, dict)
        if 'strategy' in result:
            self.assertIn('status', result['strategy'])

class TestDataValidation(unittest.TestCase):
    """数据验证测试"""
    
    def test_data_quality(self):
        """测试数据质量"""
        data_engine = DataEngine()
        result = data_engine.get_data_pipeline("AAPL", "2024-01-01", "2024-01-31")
        
        if result['status'] == 'success':
            data = result['processed_data']
            
            # 检查数据完整性
            self.assertFalse(data.empty)
            self.assertGreater(len(data), 0)
            
            # 检查价格数据合理性
            self.assertTrue((data['high'] >= data['low']).all())
            self.assertTrue((data['high'] >= data['open']).all())
            self.assertTrue((data['high'] >= data['close']).all())
            self.assertTrue((data['low'] <= data['open']).all())
            self.assertTrue((data['low'] <= data['close']).all())
            
            # 检查成交量为正数
            self.assertTrue((data['volume'] > 0).all())

class TestPerformanceMetrics(unittest.TestCase):
    """性能指标测试"""
    
    def test_backtest_metrics(self):
        """测试回测指标计算"""
        data_engine = DataEngine()
        strategy_engine = StrategyEngine(data_engine)
        
        result = strategy_engine.run_strategy_pipeline(
            "AAPL", "moving_average", "2024-01-01", "2024-02-01", 1000000
        )
        
        if result['status'] == 'success':
            metrics = result['backtest_results']
            
            # 验证夏普比率合理性
            sharpe = metrics['sharpe_ratio']
            self.assertIsInstance(sharpe, (int, float))
            self.assertGreater(sharpe, -5)  # 不应该过度负值
            self.assertLess(sharpe, 10)     # 不应该过度正值
            
            # 验证最大回撤合理性
            max_dd = metrics['max_drawdown']
            self.assertIsInstance(max_dd, (int, float))
            self.assertGreaterEqual(max_dd, 0)  # 回撤应为正值
            self.assertLessEqual(max_dd, 100)   # 不应超过100%
            
            # 验证胜率合理性
            win_rate = metrics['win_rate']
            self.assertIsInstance(win_rate, (int, float))
            self.assertGreaterEqual(win_rate, 0)
            self.assertLessEqual(win_rate, 100)

if __name__ == '__main__':
    # 运行测试套件
    unittest.main(verbosity=2) 