from typing import Optional
from datetime import datetime, timedelta
from uuid import UUID
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from .schemas import User, UserCreate
from ..database.repositories import DatabaseWrapper

class AuthService:
    def __init__(self, db: DatabaseWrapper, config: dict):
        self.db = db
        self.config = config
        self.ph = PasswordHasher()
        self.secret_key = config["jwt_secret"]
        self.token_expire_minutes = config.get("token_expire_minutes", 60)
    
    async def register_user(self, user_data: UserCreate) -> User:
        # 检查用户名和邮箱是否已存在
        existing = await self.db.fetch_one(
            "SELECT username, email FROM users WHERE username = $1 OR email = $2",
            user_data.username, user_data.email
        )
        if existing:
            raise ValueError("Username or email already exists")
        
        # 密码哈希
        password_hash = self.ph.hash(user_data.password)
        
        # 创建用户
        user_id = await self.db.fetch_one("""
            INSERT INTO users (username, email, password_hash)
            VALUES ($1, $2, $3)
            RETURNING user_id
        """, user_data.username, user_data.email, password_hash)
        
        return User(
            user_id=user_id["user_id"],
            username=user_data.username,
            email=user_data.email
        )
    
    async def authenticate(self, username: str, password: str) -> Optional[tuple[User, str]]:
        user_record = await self.db.fetch_one("""
            SELECT user_id, username, email, password_hash
            FROM users
            WHERE username = $1 AND is_active = true
        """, username)
        
        if not user_record:
            return None
        
        try:
            self.ph.verify(user_record["password_hash"], password)
        except VerifyMismatchError:
            return None
        
        # 更新最后登录时间
        await self.db.execute(
            "UPDATE users SET last_login = NOW() WHERE user_id = $1",
            user_record["user_id"]
        )
        
        user = User(
            user_id=user_record["user_id"],
            username=user_record["username"],
            email=user_record["email"]
        )
        
        token = self._create_token(user.user_id)
        return user, token
    
    def _create_token(self, user_id: UUID) -> str:
        expires = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        payload = {
            "user_id": str(user_id),
            "exp": expires
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    async def get_current_user(self, token: str) -> Optional[User]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            user_id = UUID(payload["user_id"])
        except (jwt.InvalidTokenError, ValueError):
            return None
        
        user_record = await self.db.fetch_one("""
            SELECT user_id, username, email
            FROM users
            WHERE user_id = $1 AND is_active = true
        """, user_id)
        
        if not user_record:
            return None
        
        return User(
            user_id=user_record["user_id"],
            username=user_record["username"],
            email=user_record["email"]
        )
