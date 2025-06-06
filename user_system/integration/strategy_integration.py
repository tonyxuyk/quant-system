from typing import Optional, List
from uuid import UUID
from datetime import date
from ..database.repositories import DatabaseWrapper
from ..auth.schemas import SavedBacktest

class StrategyIntegration:
    def __init__(self, db: DatabaseWrapper):
        self.db = db
    
    async def save_backtest(
        self,
        user_id: UUID,
        backtest_id: UUID,
        save_name: Optional[str] = None,
        notes: Optional[str] = None
    ) -> SavedBacktest:
        """保存回测结果"""
        save_id = await self.db.fetch_one("""
            INSERT INTO saved_backtests (
                user_id, backtest_id, save_name, notes
            ) VALUES ($1, $2, $3, $4)
            RETURNING save_id
        """, user_id, backtest_id, save_name, notes)
        
        return await self.get_saved_backtest(save_id["save_id"])
    
    async def get_saved_backtest(self, save_id: UUID) -> Optional[SavedBacktest]:
        """获取保存的回测"""
        record = await self.db.fetch_one("""
            SELECT *
            FROM saved_backtests
            WHERE save_id = $1
        """, save_id)
        
        return SavedBacktest(**record) if record else None
    
    async def get_user_backtests(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0
    ) -> List[SavedBacktest]:
        """获取用户的回测历史"""
        records = await self.db.fetch_all("""
            SELECT sb.*, b.strategy_name, b.start_date, b.end_date,
                   b.initial_capital, b.final_value, b.sharpe_ratio
            FROM saved_backtests sb
            JOIN backtests b ON sb.backtest_id = b.backtest_id
            WHERE sb.user_id = $1
            ORDER BY sb.created_at DESC
            LIMIT $2 OFFSET $3
        """, user_id, limit, offset)
        
        return [SavedBacktest(**record) for record in records]
    
    async def delete_saved_backtest(self, save_id: UUID) -> bool:
        """删除保存的回测"""
        result = await self.db.execute("""
            DELETE FROM saved_backtests
            WHERE save_id = $1
        """, save_id)
        
        return "DELETE 1" in result
    
    async def associate_backtest_with_picks(
        self,
        backtest_id: UUID,
        user_id: UUID,
        symbols: List[str]
    ) -> None:
        """关联回测结果与选股记录"""
        # 首先获取相关的选股记录
        pick_records = await self.db.fetch_all("""
            SELECT pick_id
            FROM stock_picks
            WHERE user_id = $1
            AND symbol = ANY($2)
            AND status = 'watching'
        """, user_id, symbols)
        
        # 批量更新选股记录的状态
        if pick_records:
            pick_ids = [record["pick_id"] for record in pick_records]
            await self.db.execute_many("""
                UPDATE stock_picks
                SET backtest_id = $1
                WHERE pick_id = $2
            """, [(backtest_id, pick_id) for pick_id in pick_ids])
    
    async def get_strategy_performance(
        self,
        user_id: UUID,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> dict:
        """获取策略绩效统计"""
        query = """
            SELECT
                COUNT(*) as total_backtests,
                AVG(final_value / initial_capital - 1) * 100 as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                MAX(final_value / initial_capital - 1) * 100 as best_return,
                MIN(final_value / initial_capital - 1) * 100 as worst_return
            FROM saved_backtests sb
            JOIN backtests b ON sb.backtest_id = b.backtest_id
            WHERE sb.user_id = $1
        """
        params = [user_id]
        
        if start_date:
            query += " AND b.start_date >= $2"
            params.append(start_date)
        if end_date:
            query += " AND b.end_date <= $" + str(len(params) + 1)
            params.append(end_date)
        
        stats = await self.db.fetch_one(query, *params)
        return dict(stats)
