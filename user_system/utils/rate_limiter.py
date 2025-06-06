from typing import Optional
import time
import asyncio
from dataclasses import dataclass
from user_system.database.redis_cache import RedisCache

@dataclass
class RateLimitConfig:
    requests: int  # 请求数量
    period: int   # 时间周期(秒)
    burst: Optional[int] = None  # 突发请求限制

class RateLimiter:
    def __init__(
        self,
        redis: RedisCache,
        config: RateLimitConfig
    ):
        self.redis = redis
        self.config = config
        self.burst = config.burst or config.requests
    
    async def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        current = time.time()
        period_start = current - self.config.period
        
        async with self.redis.redis.pipeline() as pipe:
            # 移除过期的请求记录
            pipe.zremrangebyscore(key, 0, period_start)
            # 获取当前周期内的请求数
            pipe.zcard(key)
            # 添加新的请求记录
            pipe.zadd(key, {str(current): current})
            # 设置过期时间
            pipe.expire(key, self.config.period)
            
            results = await pipe.execute()
        
        request_count = results[1]
        
        # 检查是否超过限制
        if request_count > self.burst:
            return False
        
        return request_count <= self.config.requests
    
    async def get_remaining(self, key: str) -> int:
        """获取剩余可用请求数"""
        current = time.time()
        period_start = current - self.config.period
        
        # 清理过期记录并获取当前数量
        async with self.redis.redis.pipeline() as pipe:
            pipe.zremrangebyscore(key, 0, period_start)
            pipe.zcard(key)
            results = await pipe.execute()
        
        current_requests = results[1]
        return max(0, self.config.requests - current_requests)
    
    async def get_reset_time(self, key: str) -> float:
        """获取限制重置时间(秒)"""
        current = time.time()
        oldest_request = await self.redis.redis.zrange(
            key, 0, 0, withscores=True
        )
        
        if not oldest_request:
            return 0
        
        return max(0, self.config.period - (current - oldest_request[0][1]))

class APIRateLimiter:
    def __init__(self, redis: RedisCache):
        self.redis = redis
        
        # 定义不同API的限流规则
        self.limits = {
            "default": RateLimitConfig(
                requests=100,
                period=60,  # 1分钟
                burst=120
            ),
            "market_data": RateLimitConfig(
                requests=300,
                period=60,
                burst=350
            ),
            "backtest": RateLimitConfig(
                requests=50,
                period=3600,  # 1小时
                burst=60
            )
        }
        
        self.limiters = {
            key: RateLimiter(redis, config)
            for key, config in self.limits.items()
        }
    
    def get_limiter(self, api_name: str) -> RateLimiter:
        """获取指定API的限流器"""
        return self.limiters.get(api_name, self.limiters["default"])
    
    async def check_rate_limit(
        self,
        api_name: str,
        user_id: str
    ) -> tuple[bool, dict]:
        """检查API访问限制
        返回: (是否允许, 限制信息)
        """
        limiter = self.get_limiter(api_name)
        key = f"rate_limit:{api_name}:{user_id}"
        
        is_allowed = await limiter.is_allowed(key)
        remaining = await limiter.get_remaining(key)
        reset_time = await limiter.get_reset_time(key)
        
        return is_allowed, {
            "remaining": remaining,
            "reset": reset_time,
            "limit": limiter.config.requests
        }
