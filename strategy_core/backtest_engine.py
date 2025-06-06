import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional
import matplotlib.pyplot as plt

class BacktestEngine:
    """多股票多周期点对点回测引擎，支持复权、滑点、详细交易明细和多指标，支持投资组合、多空、止损止盈"""
    def __init__(self, commission=0.001, tax=0.001, slippage=0.0, freq='D', adj=None):
        self.commission = commission
        self.tax = tax
        self.slippage = slippage
        self.freq = freq  # 'D', '60min', '15min', '5min', '1min', 'W', 'M'
        self.adj = adj    # None, 'qfq', 'hfq'

    def run(self, stock_data: Dict[str, pd.DataFrame],
            strategy_func: Callable[[pd.DataFrame], pd.Series],
            initial_cash: float = 100000,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            stop_loss: float = None,
            take_profit: float = None,
            portfolio_weights: Optional[Dict[str, float]] = None) -> dict:
        """
        stock_data: {code: DataFrame}, DataFrame需有DatetimeIndex和'close','open','high','low','volume'等
        strategy_func: 输入df，返回信号Series（1买-1卖0持）
        stop_loss, take_profit: 止损止盈百分比（如0.05表示5%）
        portfolio_weights: 投资组合权重，{code: weight}
        """
        all_trades = []
        equity_curve = pd.Series(dtype=float)
        cash = initial_cash
        code_equity = {}
        code_trades = {}
        if portfolio_weights is None:
            # 均分资金
            portfolio_weights = {code: 1/len(stock_data) for code in stock_data}
        for code, df in stock_data.items():
            df = df.copy()
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if self.adj in ['qfq', 'hfq'] and 'adj_factor' in df.columns:
                # 复权处理
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col] * df['adj_factor'] / df['adj_factor'].iloc[0]
            # 生成信号
            signal = strategy_func(df)
            # 单只股票分配资金
            sub_cash = initial_cash * portfolio_weights.get(code, 0)
            trades, equity = self._simulate_trades(
                df, signal, code, sub_cash, stop_loss, take_profit
            )
            all_trades.extend(trades)
            code_equity[code] = equity
            code_trades[code] = trades
            equity_curve = equity_curve.add(equity, fill_value=0)
        equity_curve = equity_curve.fillna(method='ffill').fillna(0)
        # 计算指标
        stats = self.calc_stats(equity_curve, all_trades, initial_cash)
        stats['equity_curve'] = equity_curve
        stats['trades'] = pd.DataFrame(all_trades)
        stats['code_equity'] = code_equity
        stats['code_trades'] = code_trades
        self._plot_results(equity_curve, stats)
        return stats

    def _simulate_trades(self, df, signal, code, initial_cash, stop_loss=None, take_profit=None):
        position = 0
        entry_price = 0
        entry_date = None
        cash = initial_cash
        equity_curve = pd.Series(index=df.index, dtype=float)
        trades = []
        last_price = 0
        for i, (dt, row) in enumerate(df.iterrows()):
            price = row['close'] * (1 + self.slippage * np.random.randn())
            sig = signal.iloc[i]
            # 多空双向支持
            if sig == 1 and position <= 0:
                # 平空开多
                if position < 0:
                    # 平空
                    proceeds = abs(position) * price * (1 - self.commission - self.tax)
                    cash += proceeds
                    trades.append({'date': dt, 'code': code, 'price': price, 'qty': abs(position), 'type': 'COVER', 'cash': cash})
                    position = 0
                # 开多
                qty = int(cash // (price * (1 + self.commission)))
                if qty > 0:
                    cost = qty * price * (1 + self.commission)
                    cash -= cost
                    position += qty
                    entry_price = price
                    entry_date = dt
                    trades.append({'date': dt, 'code': code, 'price': price, 'qty': qty, 'type': 'BUY', 'cash': cash})
            elif sig == -1 and position >= 0:
                # 平多开空
                if position > 0:
                    # 平多
                    proceeds = position * price * (1 - self.commission - self.tax)
                    cash += proceeds
                    trades.append({'date': dt, 'code': code, 'price': price, 'qty': position, 'type': 'SELL', 'cash': cash})
                    position = 0
                # 开空
                qty = int(cash // (price * (1 + self.commission)))
                if qty > 0:
                    proceeds = qty * price * (1 - self.commission - self.tax)
                    cash += proceeds
                    position -= qty
                    entry_price = price
                    entry_date = dt
                    trades.append({'date': dt, 'code': code, 'price': price, 'qty': qty, 'type': 'SHORT', 'cash': cash})
            # 止损止盈机制
            if position > 0 and entry_price > 0:
                if stop_loss and price <= entry_price * (1 - stop_loss):
                    proceeds = position * price * (1 - self.commission - self.tax)
                    cash += proceeds
                    trades.append({'date': dt, 'code': code, 'price': price, 'qty': position, 'type': 'SELL_STOP', 'cash': cash})
                    position = 0
                elif take_profit and price >= entry_price * (1 + take_profit):
                    proceeds = position * price * (1 - self.commission - self.tax)
                    cash += proceeds
                    trades.append({'date': dt, 'code': code, 'price': price, 'qty': position, 'type': 'SELL_TP', 'cash': cash})
                    position = 0
            elif position < 0 and entry_price > 0:
                if stop_loss and price >= entry_price * (1 + stop_loss):
                    proceeds = abs(position) * price * (1 - self.commission - self.tax)
                    cash += proceeds
                    trades.append({'date': dt, 'code': code, 'price': price, 'qty': abs(position), 'type': 'COVER_STOP', 'cash': cash})
                    position = 0
                elif take_profit and price <= entry_price * (1 - take_profit):
                    proceeds = abs(position) * price * (1 - self.commission - self.tax)
                    cash += proceeds
                    trades.append({'date': dt, 'code': code, 'price': price, 'qty': abs(position), 'type': 'COVER_TP', 'cash': cash})
                    position = 0
            equity_curve[dt] = cash + position * price
            last_price = price
        # 清仓
        if position > 0:
            cash += position * last_price * (1 - self.commission - self.tax)
            trades.append({'date': df.index[-1], 'code': code, 'price': last_price, 'qty': position, 'type': 'SELL', 'cash': cash})
            position = 0
        elif position < 0:
            cash += abs(position) * last_price * (1 - self.commission - self.tax)
            trades.append({'date': df.index[-1], 'code': code, 'price': last_price, 'qty': abs(position), 'type': 'COVER', 'cash': cash})
            position = 0
        return trades, equity_curve

    @staticmethod
    def calc_stats(equity: pd.Series, trades: List[dict], initial_cash: float) -> dict:
        ret = equity / initial_cash - 1
        ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
        daily_ret = equity.pct_change().fillna(0)
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-8) * np.sqrt(252)
        drawdown = 1 - equity / equity.cummax()
        max_drawdown = drawdown.max()
        # 索提诺比率
        downside = daily_ret[daily_ret < 0].std() + 1e-8
        sortino = daily_ret.mean() / downside * np.sqrt(252)
        # 胜率、持仓天数、盈亏比
        trade_df = pd.DataFrame(trades)
        win_rate = 0
        avg_holding = 0
        profit_loss_ratio = 0
        if not trade_df.empty:
            buy_prices = trade_df[trade_df['type'].isin(['BUY', 'SHORT'])]['price'].values
            sell_prices = trade_df[trade_df['type'].str.contains('SELL|COVER')]['price'].values
            n = min(len(buy_prices), len(sell_prices))
            win_rate = np.mean(sell_prices[:n] > buy_prices[:n]) if n > 0 else 0
            # 持仓天数
            if 'date' in trade_df.columns:
                holding_days = []
                entries = trade_df[trade_df['type'].isin(['BUY', 'SHORT'])]
                exits = trade_df[trade_df['type'].str.contains('SELL|COVER')]
                for entry, exit in zip(entries['date'], exits['date']):
                    holding_days.append((exit - entry).days if hasattr(exit, 'days') else 1)
                avg_holding = np.mean(holding_days) if holding_days else 0
            # 盈亏比
            profits = []
            losses = []
            for i in range(n):
                pnl = sell_prices[i] - buy_prices[i]
                if pnl > 0:
                    profits.append(pnl)
                elif pnl < 0:
                    losses.append(-pnl)
            profit_loss_ratio = (np.mean(profits) / np.mean(losses)) if profits and losses else 0
        return {
            'total_return': ret.iloc[-1],
            'annual_return': ann_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'sortino': sortino,
            'win_rate': win_rate,
            'avg_holding_days': avg_holding,
            'profit_loss_ratio': profit_loss_ratio
        }

    def _plot_results(self, equity_curve, stats):
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label='Equity Curve')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Account Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        # 回撤曲线
        drawdown = 1 - equity_curve / equity_curve.cummax()
        plt.figure(figsize=(12, 3))
        plt.plot(drawdown, label='Drawdown')
        plt.title('Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        plt.show()