from typing import List, Optional
from uuid import UUID
from datetime import date, datetime
import asyncio
from ..database.repositories import DatabaseWrapper
from ..auth.schemas import Position

class PositionService:
    def __init__(self, db: DatabaseWrapper, market_data_client):
        self.db = db
        self.market = market_data_client
        self._cache = {}
        self._cache_time = {}
        self.CACHE_TTL = 300  # 5 minutes
        
    async def open_position(
        self,
        user_id: UUID,
        symbol: str,
        entry_date: date,
        entry_price: float,
        quantity: float,
        strategy_id: Optional[UUID] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Position:
        """开仓操作"""
        position_id = await self.db.fetch_one("""
            INSERT INTO user_positions (
                user_id, symbol, entry_date, entry_price, quantity,
                strategy_id, stop_loss, take_profit, notes
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING position_id
        """, user_id, symbol, entry_date, entry_price, quantity,
             strategy_id, stop_loss, take_profit, notes)
        
        return await self.get_position(position_id["position_id"])
    
    async def close_position(
        self,
        position_id: UUID,
        exit_date: date,
        exit_price: float
    ) -> Position:
        """平仓操作"""
        await self.db.execute("""
            UPDATE user_positions
            SET exit_date = $1,
                exit_price = $2,
                is_active = false
            WHERE position_id = $3
        """, exit_date, exit_price, position_id)
        
        return await self.get_position(position_id)
    
    async def get_position(self, position_id: UUID) -> Optional[Position]:
        """获取单个持仓详情"""
        record = await self.db.fetch_one("""
            SELECT *
            FROM user_positions
            WHERE position_id = $1
        """, position_id)
        
        return Position(**record) if record else None
    
    async def get_current_positions(
        self,
        user_id: UUID,
        include_market_value: bool = True
    ) -> List[Position]:
        """获取当前持仓列表"""
        records = await self.db.fetch_all("""
            SELECT *
            FROM user_positions
            WHERE user_id = $1 AND is_active = true
            ORDER BY entry_date DESC
        """, user_id)
        
        positions = [Position(**record) for record in records]
        
        if include_market_value:
            # 并行获取实时市场数据
            tasks = []
            for position in positions:
                tasks.append(self._get_market_value(position.symbol))
            market_values = await asyncio.gather(*tasks)
            
            # 更新市场价值
            for position, market_value in zip(positions, market_values):
                position.current_price = market_value
                position.market_value = market_value * position.quantity
                if position.entry_price:
                    position.unrealized_gain = (
                        (market_value - position.entry_price) 
                        / position.entry_price 
                        * 100
                    )
        
        return positions
    
    async def _get_market_value(self, symbol: str) -> float:
        """获取实时市场价格(带缓存)"""
        now = datetime.now().timestamp()
        if (
            symbol in self._cache 
            and now - self._cache_time.get(symbol, 0) < self.CACHE_TTL
        ):
            return self._cache[symbol]
        
        price = await self.market.get_latest_price(symbol)
        self._cache[symbol] = price
        self._cache_time[symbol] = now
        return price
    
    async def check_stop_loss_take_profit(self) -> List[Position]:
        """检查止盈止损
        返回触发止盈止损的持仓列表"""
        positions = await self.db.fetch_all("""
            SELECT *
            FROM user_positions
            WHERE is_active = true
            AND (stop_loss IS NOT NULL OR take_profit IS NOT NULL)
        """)
        
        triggered = []
        for pos in positions:
            current_price = await self._get_market_value(pos["symbol"])
            
            if (
                pos["stop_loss"] 
                and current_price <= pos["stop_loss"]
            ) or (
                pos["take_profit"] 
                and current_price >= pos["take_profit"]
            ):
                triggered.append(Position(**pos))
        
        return triggered
