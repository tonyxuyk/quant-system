-- V1__create_user_tables.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE user_preferences (
    preference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    default_strategy_id UUID,
    risk_level VARCHAR(20),
    notification_settings JSONB,
    UNIQUE(user_id)
);

CREATE TABLE stock_picks (
    pick_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    pick_date DATE NOT NULL,
    reason TEXT,
    expected_gain DECIMAL(5,2),
    actual_gain DECIMAL(5,2),
    status VARCHAR(20) CHECK (status IN ('watching', 'holding', 'sold', 'expired')),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE user_positions (
    position_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    quantity DECIMAL(12,4) NOT NULL,
    exit_date DATE,
    exit_price DECIMAL(12,4),
    stop_loss DECIMAL(12,4),
    take_profit DECIMAL(12,4),
    strategy_id UUID,
    notes TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE saved_backtests (
    save_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    backtest_id UUID,
    save_name VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 创建触发器以自动更新updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_positions_updated_at
    BEFORE UPDATE ON user_positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
