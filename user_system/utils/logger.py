import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
from uuid import UUID

class AuditLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 设置审计日志
        audit_handler = logging.FileHandler(
            self.log_dir / "audit.log"
        )
        audit_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.addHandler(audit_handler)
        
        # 设置错误日志
        error_handler = logging.FileHandler(
            self.log_dir / "error.log"
        )
        error_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
                'Exception: %(exc_info)s\n'
                'Stack Trace: %(stack_info)s\n'
            )
        )
        
        self.error_logger = logging.getLogger("error")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.addHandler(error_handler)
    
    def log_user_action(
        self,
        user_id: UUID,
        action: str,
        target: str,
        details: Optional[dict] = None
    ):
        """记录用户操作"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": str(user_id),
            "action": action,
            "target": target,
            "details": details or {}
        }
        
        self.audit_logger.info(json.dumps(log_entry))
    
    def log_system_error(
        self,
        error: Exception,
        context: Optional[dict] = None
    ):
        """记录系统错误"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.error_logger.error(
            json.dumps(error_entry),
            exc_info=True,
            stack_info=True
        )
    
    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[UUID] = None,
        details: Optional[dict] = None
    ):
        """记录安全相关事件"""
        security_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": str(user_id) if user_id else None,
            "details": details or {}
        }
        
        self.audit_logger.warning(json.dumps(security_entry))
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[dict] = None
    ):
        """记录性能指标"""
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "value": value,
            "tags": tags or {}
        }
        
        self.audit_logger.info(
            f"METRIC: {json.dumps(metric_entry)}"
        )

# 创建单例实例
logger = AuditLogger()
