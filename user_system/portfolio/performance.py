from typing import List, Dict, Optional
from uuid import UUID
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from ..database.repositories import DatabaseWrapper

class PerformanceService:
    def __init__(self, db: DatabaseWrapper):
        self.db = db
    
    async def calculate_portfolio_performance(
        self,
        user_id: UUID,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict:
        """计算投资组合绩效"""
        # 获取平仓记录
        closed_positions = await self.db.fetch_all("""
            SELECT *
            FROM user_positions
            WHERE user_id = $1
            AND is_active = false
            AND exit_date IS NOT NULL
            AND exit_price IS NOT NULL
            AND ($2::date IS NULL OR entry_date >= $2)
            AND ($3::date IS NULL OR exit_date <= $3)
        """, user_id, start_date, end_date)
        
        if not closed_positions:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_return": 0,
                "max_return": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
        
        # 转换为DataFrame进行分析
        df = pd.DataFrame(closed_positions)
        
        # 计算每笔交易的收益率
        df["return"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100
        
        # 计算关键指标
        total_trades = len(df)
        winning_trades = len(df[df["return"] > 0])
        win_rate = winning_trades / total_trades * 100
        avg_return = df["return"].mean()
        max_return = df["return"].max()
        min_return = df["return"].min()
        
        # 计算夏普比率
        returns_std = df["return"].std()
        risk_free_rate = 2.0  # 假设无风险利率为2%
        sharpe_ratio = (avg_return - risk_free_rate) / returns_std if returns_std > 0 else 0
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(df)
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "max_return": max_return,
            "min_return": min_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """计算最大回撤"""
        df = df.sort_values("entry_date")
        cumulative = (1 + df["return"] / 100).cumprod()
        rolling_max = cumulative.expanding(min_periods=1).max()
        drawdowns = (cumulative - rolling_max) / rolling_max * 100
        return abs(drawdowns.min())
    
    async def get_portfolio_history(
        self,
        user_id: UUID,
        interval: str = "daily"
    ) -> List[Dict]:
        """获取投资组合历史数据
        interval: daily, weekly, monthly"""
        positions = await self.db.fetch_all("""
            SELECT *
            FROM user_positions
            WHERE user_id = $1
            ORDER BY entry_date
        """, user_id)
        
        if not positions:
            return []
        
        df = pd.DataFrame(positions)
        
        # 设置时间范围
        start_date = df["entry_date"].min()
        end_date = datetime.now().date()
        dates = pd.date_range(start_date, end_date, freq=interval[0].upper())
        
        portfolio_values = []
        for current_date in dates:
            # 获取当前持仓
            current_positions = df[
                (df["entry_date"] <= current_date) &
                ((df["exit_date"].isna()) | (df["exit_date"] > current_date))
            ]
            
            if len(current_positions) > 0:
                total_value = (
                    current_positions["quantity"] * current_positions["entry_price"]
                ).sum()
                
                portfolio_values.append({
                    "date": current_date.date(),
                    "value": float(total_value),
                    "positions": len(current_positions)
                })
        
        return portfolio_values
    
    async def get_sector_allocation(self, user_id: UUID) -> List[Dict]:
        """获取行业配置分析"""
        positions = await self.db.fetch_all("""
            SELECT p.*, s.sector
            FROM user_positions p
            LEFT JOIN stock_info s ON p.symbol = s.symbol
            WHERE p.user_id = $1 AND p.is_active = true
        """, user_id)
        
        if not positions:
            return []
        
        df = pd.DataFrame(positions)
        df["market_value"] = df["quantity"] * df["entry_price"]
        sector_allocation = df.groupby("sector")["market_value"].sum()
        total_value = sector_allocation.sum()
        
        return [
            {
                "sector": sector,
                "value": float(value),
                "percentage": float(value / total_value * 100)
            }
            for sector, value in sector_allocation.items()
        ]
    
    async def get_risk_metrics(self, user_id: UUID) -> Dict:
        """获取风险指标"""
        # 获取所有平仓记录的收益率数据
        returns = await self.db.fetch_all("""
            SELECT
                (exit_price - entry_price) / entry_price as return_rate
            FROM user_positions
            WHERE user_id = $1
            AND is_active = false
            AND exit_date IS NOT NULL
        """, user_id)
        
        if not returns:
            return {
                "volatility": 0,
                "var_95": 0,
                "expected_shortfall": 0,
                "beta": 0
            }
        
        returns_series = pd.Series([r["return_rate"] for r in returns])
        
        # 计算波动率（年化）
        volatility = returns_series.std() * np.sqrt(252)
        
        # 计算95% VaR
        var_95 = np.percentile(returns_series, 5)
        
        # 计算预期亏空
        expected_shortfall = returns_series[returns_series <= var_95].mean()
        
        # TODO: 添加对Beta的计算（需要市场指数数据）
        
        return {
            "volatility": float(volatility),
            "var_95": float(var_95),
            "expected_shortfall": float(expected_shortfall),
            "beta": None  # 待实现
        }
