from celery import Celery
from typing import List, Dict
from uuid import UUID
from datetime import datetime
import pandas as pd
import numpy as np

# 创建Celery实例
celery_app = Celery(
    'quant_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Celery配置
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True
)

@celery_app.task
def calculate_portfolio_metrics(
    positions: List[Dict],
    start_date: str,
    end_date: str
) -> Dict:
    """计算投资组合指标的异步任务"""
    df = pd.DataFrame(positions)
    
    # 计算收益率序列
    df['return'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
    
    # 基础指标
    total_trades = len(df)
    winning_trades = len(df[df['return'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 收益指标
    total_return = df['return'].sum()
    avg_return = df['return'].mean()
    std_return = df['return'].std()
    
    # 风险指标
    sharpe_ratio = (avg_return - 0.02) / std_return if std_return > 0 else 0
    max_drawdown = calculate_max_drawdown(df)
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'total_return': total_return * 100,
        'avg_return': avg_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'volatility': std_return * np.sqrt(252) * 100
    }

@celery_app.task
def analyze_stock_picks(picks: List[Dict]) -> Dict:
    """分析选股表现的异步任务"""
    df = pd.DataFrame(picks)
    
    # 按状态分组统计
    status_stats = df['status'].value_counts().to_dict()
    
    # 计算成功率
    closed_picks = df[df['status'].isin(['sold', 'expired'])]
    success_rate = (
        len(closed_picks[closed_picks['actual_gain'] > 0]) / 
        len(closed_picks) if len(closed_picks) > 0 else 0
    )
    
    # 收益统计
    gain_stats = {
        'avg_gain': closed_picks['actual_gain'].mean(),
        'max_gain': closed_picks['actual_gain'].max(),
        'min_gain': closed_picks['actual_gain'].min()
    }
    
    # 按月统计选股数量
    df['month'] = pd.to_datetime(df['pick_date']).dt.to_period('M')
    monthly_picks = df.groupby('month').size().to_dict()
    
    return {
        'status_distribution': status_stats,
        'success_rate': success_rate * 100,
        'gain_stats': gain_stats,
        'monthly_picks': {str(k): v for k, v in monthly_picks.items()}
    }

@celery_app.task
def run_batch_backtests(
    strategy_ids: List[UUID],
    start_date: str,
    end_date: str,
    initial_capital: float
) -> List[Dict]:
    """批量执行回测的异步任务"""
    results = []
    for strategy_id in strategy_ids:
        # 这里需要集成实际的回测逻辑
        result = run_single_backtest(
            strategy_id,
            start_date,
            end_date,
            initial_capital
        )
        results.append(result)
    return results

def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """计算最大回撤"""
    df = df.sort_values('entry_date')
    cumulative = (1 + df['return']).cumprod()
    rolling_max = cumulative.expanding(min_periods=1).max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return abs(drawdowns.min()) if len(drawdowns) > 0 else 0

def run_single_backtest(
    strategy_id: UUID,
    start_date: str,
    end_date: str,
    initial_capital: float
) -> Dict:
    """执行单个回测"""
    # 这里需要实现实际的回测逻辑
    pass
