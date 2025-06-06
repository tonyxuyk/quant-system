from typing import Optional, List, Dict, Any
from uuid import UUID
import asyncpg
from datetime import date, datetime
from dataclasses import dataclass
import json

@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    min_size: int = 10
    max_size: int = 20

class DatabaseWrapper:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self._cache = {}
    
    @classmethod
    async def create(cls, config: DatabaseConfig) -> 'DatabaseWrapper':
        pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database=config.database,
            min_size=config.min_size,
            max_size=config.max_size,
            statement_cache_size=1000,
            max_cached_statement_lifetime=300,  # 5 minutes
            command_timeout=60.0  # 1 minute timeout
        )
        
        # 注册JSON编码器
        await pool.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
        
        return cls(pool)
    
    def _generate_cache_key(self, query: str, args: tuple) -> str:
        """生成缓存键"""
        return f"{query}:{hash(args)}"
    
    async def fetch_one_cached(
        self,
        query: str,
        *args,
        ttl: int = 300  # 5 minutes default
    ) -> Optional[asyncpg.Record]:
        """带缓存的单行查询"""
        cache_key = self._generate_cache_key(query, args)
        
        # 检查缓存
        cached = self._cache.get(cache_key)
        if cached and cached['expires'] > datetime.now().timestamp():
            return cached['data']
        
        # 执行查询
        result = await self.fetch_one(query, *args)
        
        # 更新缓存
        if result:
            self._cache[cache_key] = {
                'data': result,
                'expires': datetime.now().timestamp() + ttl
            }
        
        return result
    
    async def fetch_one(self, query: str, *args) -> Optional[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetch_all(self, query: str, *args) -> List[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def execute(self, query: str, *args) -> str:
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def execute_many(self, query: str, args_list: List[tuple]) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(query, args_list)
    
    async def transaction(self) -> asyncpg.transaction.Transaction:
        return self.pool.acquire()
