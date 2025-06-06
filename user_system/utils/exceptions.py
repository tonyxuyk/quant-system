from typing import Optional, Dict, Any
from uuid import UUID

class QuantSystemError(Exception):
    """量化系统基础异常类"""
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class AuthenticationError(QuantSystemError):
    """认证相关错误"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTH_ERROR",
            details=details
        )

class PermissionError(QuantSystemError):
    """权限相关错误"""
    def __init__(
        self,
        message: str,
        user_id: UUID,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "user_id": str(user_id),
            "resource_type": resource_type,
            "resource_id": resource_id
        })
        super().__init__(
            message=message,
            error_code="PERMISSION_ERROR",
            details=details
        )

class ValidationError(QuantSystemError):
    """数据验证错误"""
    def __init__(
        self,
        message: str,
        field: str,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["field"] = field
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )

class ResourceNotFoundError(QuantSystemError):
    """资源不存在错误"""
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"{resource_type} with id {resource_id} not found"
        details = details or {}
        details.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details=details
        )

class RateLimitExceededError(QuantSystemError):
    """API限流错误"""
    def __init__(
        self,
        api_name: str,
        limit_info: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"Rate limit exceeded for {api_name}"
        details = details or {}
        details.update({
            "api_name": api_name,
            "limit_info": limit_info
        })
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )

class DatabaseError(QuantSystemError):
    """数据库操作错误"""
    def __init__(
        self,
        message: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["operation"] = operation
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details
        )

class MarketDataError(QuantSystemError):
    """市场数据错误"""
    def __init__(
        self,
        message: str,
        symbol: str,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["symbol"] = symbol
        super().__init__(
            message=message,
            error_code="MARKET_DATA_ERROR",
            details=details
        )

class BacktestError(QuantSystemError):
    """回测相关错误"""
    def __init__(
        self,
        message: str,
        strategy_id: UUID,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["strategy_id"] = str(strategy_id)
        super().__init__(
            message=message,
            error_code="BACKTEST_ERROR",
            details=details
        )

class ConfigurationError(QuantSystemError):
    """配置相关错误"""
    def __init__(
        self,
        message: str,
        config_key: str,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["config_key"] = config_key
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details=details
        )
