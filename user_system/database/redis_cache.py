from typing import Optional, Any
import json
import aioredis
from datetime import datetime, timedelta

class RedisCache:
    def __init__(self, redis: aioredis.Redis):
        self.redis = redis
    
    @classmethod
    async def create(
        cls,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ) -> 'RedisCache':
        redis = await aioredis.create_redis_pool(
            f'redis://{host}:{port}',
            db=db,
            password=password,
            encoding='utf-8'
        )
        return cls(redis)
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        expire: int = 300  # 5 minutes default
    ) -> bool:
        """设置缓存值"""
        try:
            await self.redis.set(
                key,
                json.dumps(value),
                expire=expire
            )
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        return await self.redis.delete(key) > 0
    
    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的所有缓存"""
        keys = await self.redis.keys(pattern)
        if keys:
            return await self.redis.delete(*keys)
        return 0
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """增加计数器"""
        return await self.redis.incrby(key, amount)
    
    async def expire_at(self, key: str, timestamp: datetime) -> bool:
        """设置过期时间点"""
        return await self.redis.expireat(key, int(timestamp.timestamp()))
    
    async def close(self):
        """关闭连接"""
        self.redis.close()
        await self.redis.wait_closed()
    
    async def health_check(self) -> bool:
        """检查Redis连接状态"""
        try:
            return await self.redis.ping()
        except Exception:
            return False
