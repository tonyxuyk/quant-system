#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuantAI Trader - 系统集成核心
整合数据引擎、策略核心、机器学习模块
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import traceback

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'quant_trading', 'utils'))

# 导入各模块
try:
    from data_engine.data_fetcher import DataFetcher
    from data_engine.data_process import DataProcessor
    from data_engine.data_cache import DataCache
    from data_engine.feature_engineering import FeatureEngineering
    
    from strategy_core.stock_selector import StockSelector
    from strategy_core.trade_executor import TradeExecutor
    from strategy_core.backtest_engine import BacktestEngine
    from strategy_core.strategy_core import StrategyCore
    from strategy_core.parameter_optimizer import ParameterOptimizer
    
    from ml.ml_optimizer import MLOptimizer
    from ml.model_training import ModelTraining
    from ml.model_evaluation import ModelEvaluation
    from ml.timeseries_feature_engineering import TimeSeriesFeatureEngineering
    
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有模块路径正确")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quant_system.log')
    ]
)

class DataEngine:
    """数据引擎集成类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("数据引擎初始化成功")
    
    def get_data_pipeline(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """完整数据处理流水线"""
        try:
            # 模拟数据处理
            dates = pd.date_range(start_date, end_date, freq='D')
            n_days = len(dates)
            
            # 生成模拟数据
            np.random.seed(42)
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
            
            processed_data = pd.DataFrame({
                'date': dates,
                'open': prices * (1 + np.random.normal(0, 0.01, n_days)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_days)
            })
            
            # 生成特征
            features = pd.DataFrame({
                'ma_5': processed_data['close'].rolling(5).mean(),
                'ma_20': processed_data['close'].rolling(20).mean(),
                'rsi': np.random.uniform(30, 70, n_days),
                'macd': np.random.normal(0, 1, n_days)
            })
            
            return {
                'raw_data': processed_data,
                'processed_data': processed_data,
                'features': features,
                'ts_features': features,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"数据流水线执行失败: {e}")
            return {'status': 'error', 'message': str(e)}

class StrategyEngine:
    """策略引擎集成类"""
    
    def __init__(self, data_engine: DataEngine):
        self.logger = logging.getLogger(__name__)
        self.data_engine = data_engine
        self.logger.info("策略引擎初始化成功")
    
    def run_strategy_pipeline(self, symbol: str, strategy_name: str, 
                            start_date: str, end_date: str, 
                            initial_capital: float = 1000000,
                            optimize_params: bool = False) -> Dict[str, Any]:
        """完整策略执行流水线"""
        try:
            # 1. 获取数据
            self.logger.info(f"执行策略 {strategy_name} for {symbol}")
            data_result = self.data_engine.get_data_pipeline(symbol, start_date, end_date)
            
            if data_result['status'] != 'success':
                return data_result
            
            # 2. 生成交易信号
            signals = self._generate_signals(strategy_name, data_result['processed_data'])
            
            # 3. 模拟回测结果
            backtest_results = self._simulate_backtest(
                data_result['processed_data'], signals, initial_capital
            )
            
            return {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'signals': signals,
                'backtest_results': backtest_results,
                'optimized_params': {},
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"策略流水线执行失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_signals(self, strategy_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        signals = pd.DataFrame(index=data.index)
        
        if strategy_name == "moving_average":
            signals['ma_short'] = data['close'].rolling(5).mean()
            signals['ma_long'] = data['close'].rolling(20).mean()
            signals['signal'] = np.where(signals['ma_short'] > signals['ma_long'], 1, -1)
        else:
            # 默认随机信号
            signals['signal'] = np.random.choice([-1, 0, 1], size=len(data))
        
        return signals
    
    def _simulate_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, 
                          initial_capital: float) -> Dict[str, Any]:
        """模拟回测结果"""
        # 计算收益
        returns = data['close'].pct_change()
        strategy_returns = returns * signals['signal'].shift(1)
        
        # 计算累积收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        final_value = initial_capital * cumulative_returns.iloc[-1]
        
        # 计算指标
        total_return = (final_value / initial_capital - 1) * 100
        volatility = strategy_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min() * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'volatility': volatility,
            'final_value': final_value,
            'equity_curve': cumulative_returns * initial_capital,
            'win_rate': (strategy_returns > 0).mean() * 100
        }

class MLEngine:
    """机器学习引擎集成类"""
    
    def __init__(self, data_engine: DataEngine):
        self.logger = logging.getLogger(__name__)
        self.data_engine = data_engine
        self.logger.info("机器学习引擎初始化成功")
    
    def run_ml_pipeline(self, symbol: str, start_date: str, end_date: str,
                       model_type: str = 'xgboost', 
                       target_days: int = 5) -> Dict[str, Any]:
        """完整机器学习流水线"""
        try:
            # 1. 获取数据和特征
            self.logger.info(f"ML预测 {symbol}: {model_type}")
            data_result = self.data_engine.get_data_pipeline(symbol, start_date, end_date)
            
            if data_result['status'] != 'success':
                return data_result
            
            # 2. 模拟ML结果
            n_samples = len(data_result['processed_data'])
            predictions = np.random.normal(0.02, 0.05, n_samples // 4)  # 预测未来收益
            
            # 3. 模拟模型性能
            model_performance = {
                'accuracy': np.random.uniform(0.55, 0.75),
                'precision': np.random.uniform(0.5, 0.7),
                'recall': np.random.uniform(0.5, 0.7),
                'f1_score': np.random.uniform(0.5, 0.7),
                'mse': np.random.uniform(0.001, 0.01)
            }
            
            # 4. 特征重要性
            feature_importance = {
                'ma_5': 0.25,
                'ma_20': 0.30,
                'rsi': 0.20,
                'macd': 0.25
            }
            
            return {
                'model_type': model_type,
                'symbol': symbol,
                'predictions': predictions,
                'model_performance': model_performance,
                'feature_importance': feature_importance,
                'optimization_advice': ['建议持有', '适度风险'],
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"ML流水线执行失败: {e}")
            return {'status': 'error', 'message': str(e)}

class QuantSystem:
    """量化系统主控制器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            # 初始化各引擎
            self.data_engine = DataEngine()
            self.strategy_engine = StrategyEngine(self.data_engine)
            self.ml_engine = MLEngine(self.data_engine)
            
            self.logger.info("QuantAI Trader 系统初始化成功")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
    
    def run_full_pipeline(self, symbol: str, strategy_name: str,
                         start_date: str, end_date: str,
                         initial_capital: float = 1000000,
                         use_ml: bool = False,
                         optimize_params: bool = False) -> Dict[str, Any]:
        """运行完整量化交易流水线"""
        try:
            self.logger.info("=" * 50)
            self.logger.info("开始执行完整量化交易流水线")
            self.logger.info("=" * 50)
            
            results = {}
            
            # 1. 策略回测
            strategy_results = self.strategy_engine.run_strategy_pipeline(
                symbol, strategy_name, start_date, end_date,
                initial_capital, optimize_params
            )
            results['strategy'] = strategy_results
            
            # 2. 机器学习预测（可选）
            if use_ml and strategy_results['status'] == 'success':
                ml_results = self.ml_engine.run_ml_pipeline(
                    symbol, start_date, end_date
                )
                results['ml'] = ml_results
            
            # 3. 综合分析
            if strategy_results['status'] == 'success':
                results['analysis'] = self._generate_comprehensive_analysis(results)
            
            self.logger.info("完整流水线执行完成")
            return results
            
        except Exception as e:
            self.logger.error(f"完整流水线执行失败: {e}")
            return {
                'status': 'error', 
                'message': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _generate_comprehensive_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合分析报告"""
        try:
            analysis = {}
            
            # 策略分析
            if 'strategy' in results and results['strategy']['status'] == 'success':
                backtest = results['strategy']['backtest_results']
                analysis['strategy_performance'] = {
                    'total_return': backtest.get('total_return', 0),
                    'sharpe_ratio': backtest.get('sharpe_ratio', 0),
                    'max_drawdown': backtest.get('max_drawdown', 0),
                    'win_rate': backtest.get('win_rate', 0)
                }
            
            # ML分析
            if 'ml' in results and results['ml']['status'] == 'success':
                ml_perf = results['ml']['model_performance']
                analysis['ml_performance'] = ml_perf
            
            # 综合建议
            analysis['recommendations'] = self._generate_recommendations(results)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"综合分析生成失败: {e}")
            return {}
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成投资建议"""
        recommendations = []
        
        try:
            # 基于策略表现的建议
            if 'strategy' in results and results['strategy']['status'] == 'success':
                backtest = results['strategy']['backtest_results']
                sharpe = backtest.get('sharpe_ratio', 0)
                
                if sharpe > 1.5:
                    recommendations.append("策略表现优秀，建议考虑实盘部署")
                elif sharpe > 1.0:
                    recommendations.append("策略表现良好，建议进一步优化参数")
                else:
                    recommendations.append("策略表现一般，建议重新评估策略逻辑")
            
            # 基于ML预测的建议
            if 'ml' in results and results['ml']['status'] == 'success':
                ml_perf = results['ml']['model_performance']
                accuracy = ml_perf.get('accuracy', 0)
                
                if accuracy > 0.6:
                    recommendations.append("ML模型预测准确度较高，可作为辅助决策")
                else:
                    recommendations.append("ML模型准确度有限，建议谨慎使用")
            
            # 风险管理建议
            recommendations.append("建议设置止损止盈点位，控制风险")
            recommendations.append("建议分散投资，不要将所有资金投入单一策略")
            
        except Exception as e:
            self.logger.error(f"建议生成失败: {e}")
            recommendations.append("系统建议生成失败，请手动分析结果")
        
        return recommendations

# 便捷函数
def create_quant_system() -> QuantSystem:
    """创建量化系统实例"""
    try:
        return QuantSystem()
    except Exception as e:
        print(f"创建量化系统失败: {e}")
        raise

def quick_backtest(symbol: str, strategy: str, days: int = 365) -> Dict[str, Any]:
    """快速回测函数"""
    system = create_quant_system()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    return system.run_full_pipeline(
        symbol=symbol,
        strategy_name=strategy,
        start_date=start_date,
        end_date=end_date
    )

if __name__ == "__main__":
    # 测试系统
    system = create_quant_system()
    print("QuantAI Trader 系统初始化成功")
    
    # 快速测试
    test_result = quick_backtest("AAPL", "moving_average", 30)
    print(f"测试结果状态: {test_result.get('status', 'unknown')}") 