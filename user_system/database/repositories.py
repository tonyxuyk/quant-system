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
    
    @classmethod
    async def create(cls, config: DatabaseConfig) -> 'DatabaseWrapper':
        pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database=config.database,
            min_size=config.min_size,
            max_size=config.max_size
        )
        return cls(pool)
    
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
