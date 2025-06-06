from typing import Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr, constr

class UserBase(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    password: constr(min_length=8)

class User(UserBase):
    user_id: UUID
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

    class Config:
        from_attributes = True

class UserPreferences(BaseModel):
    user_id: UUID
    default_strategy_id: Optional[UUID] = None
    risk_level: Optional[str] = None
    notification_settings: dict = {}

class StockPick(BaseModel):
    pick_id: UUID
    user_id: UUID
    symbol: str
    pick_date: datetime
    reason: Optional[str]
    expected_gain: Optional[float]
    actual_gain: Optional[float]
    status: str
    notes: Optional[str]
    created_at: datetime

class Position(BaseModel):
    position_id: UUID
    user_id: UUID
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: float
    exit_date: Optional[datetime]
    exit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy_id: Optional[UUID]
    notes: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

class SavedBacktest(BaseModel):
    save_id: UUID
    user_id: UUID
    backtest_id: UUID
    save_name: Optional[str]
    notes: Optional[str]
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
