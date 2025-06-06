from typing import List, Optional
from uuid import UUID
from datetime import date
from ..database.repositories import DatabaseWrapper
from ..auth.schemas import StockPick

class PickService:
    def __init__(self, db: DatabaseWrapper):
        self.db = db
    
    async def create_pick(
        self,
        user_id: UUID,
        symbol: str,
        pick_date: date,
        reason: Optional[str] = None,
        expected_gain: Optional[float] = None,
        notes: Optional[str] = None
    ) -> StockPick:
        """创建选股记录"""
        pick_id = await self.db.fetch_one("""
            INSERT INTO stock_picks (
                user_id, symbol, pick_date, reason,
                expected_gain, status, notes
            ) VALUES ($1, $2, $3, $4, $5, 'watching', $6)
            RETURNING pick_id
        """, user_id, symbol, pick_date, reason, expected_gain, notes)
        
        return await self.get_pick(pick_id["pick_id"])
    
    async def get_pick(self, pick_id: UUID) -> Optional[StockPick]:
        """获取单个选股记录"""
        record = await self.db.fetch_one("""
            SELECT *
            FROM stock_picks
            WHERE pick_id = $1
        """, pick_id)
        
        return StockPick(**record) if record else None
    
    async def get_user_picks(
        self,
        user_id: UUID,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[StockPick]:
        """获取用户的选股记录列表"""
        query = """
            SELECT *
            FROM stock_picks
            WHERE user_id = $1
        """
        params = [user_id]
        
        if status:
            query += " AND status = $2"
            params.append(status)
        
        query += """
            ORDER BY pick_date DESC
            LIMIT ${}
            OFFSET ${}
        """.format(len(params) + 1, len(params) + 2)
        params.extend([limit, offset])
        
        records = await self.db.fetch_all(query, *params)
        return [StockPick(**record) for record in records]
    
    async def update_pick_status(
        self,
        pick_id: UUID,
        new_status: str,
        actual_gain: Optional[float] = None
    ) -> StockPick:
        """更新选股状态"""
        # 验证状态转换
        current = await self.get_pick(pick_id)
        if not current:
            raise ValueError("Pick not found")
        
        if not self._is_valid_status_transition(current.status, new_status):
            raise ValueError(f"Invalid status transition: {current.status} -> {new_status}")
        
        await self.db.execute("""
            UPDATE stock_picks
            SET status = $1,
                actual_gain = $2
            WHERE pick_id = $3
        """, new_status, actual_gain, pick_id)
        
        return await self.get_pick(pick_id)
    
    def _is_valid_status_transition(self, old_status: str, new_status: str) -> bool:
        """验证状态转换的有效性"""
        valid_transitions = {
            'watching': {'holding', 'expired'},
            'holding': {'sold'},
            'sold': set(),  # 终态
            'expired': set()  # 终态
        }
        return new_status in valid_transitions.get(old_status, set())
    
    async def search_picks(
        self,
        user_id: UUID,
        query: str,
        limit: int = 20
    ) -> List[StockPick]:
        """搜索选股记录"""
        records = await self.db.fetch_all("""
            SELECT *
            FROM stock_picks
            WHERE user_id = $1
            AND (
                symbol ILIKE $2
                OR reason ILIKE $2
                OR notes ILIKE $2
            )
            ORDER BY pick_date DESC
            LIMIT $3
        """, user_id, f"%{query}%", limit)
        
        return [StockPick(**record) for record in records]
    
    async def get_pick_performance(self, user_id: UUID) -> dict:
        """获取选股绩效统计"""
        stats = await self.db.fetch_one("""
            SELECT
                COUNT(*) as total_picks,
                COUNT(CASE WHEN status = 'sold' THEN 1 END) as closed_picks,
                AVG(CASE WHEN status = 'sold' THEN actual_gain END) as avg_gain,
                MAX(CASE WHEN status = 'sold' THEN actual_gain END) as max_gain,
                MIN(CASE WHEN status = 'sold' THEN actual_gain END) as max_loss
            FROM stock_picks
            WHERE user_id = $1
        """, user_id)
        
        return dict(stats)
