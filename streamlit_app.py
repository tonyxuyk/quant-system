#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tony&Associates QuantAI Trader - Streamlit Cloud版本
量化交易系统Web界面 - 修复版本
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# 页面配置
st.set_page_config(
    page_title="Tony&Associates QuantAI Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.success-banner {
    background: #4CAF50;
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    text-align: center;
}
.error-banner {
    background: #f44336;
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    text-align: center;
}
.market-badge {
    padding: 0.2rem 0.5rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
}
.market-a { background: #ff6b6b; color: white; }
.market-hk { background: #4ecdc4; color: white; }
.market-us { background: #45b7d1; color: white; }
</style>
""", unsafe_allow_html=True)

# 导入核心模块（带错误处理）
try:
    from integration import create_quant_system
    from strategy_core.backtest_engine import BacktestEngine
    from strategy_core.strategy_core import (
        MovingAverageCrossStrategy, RSIStrategy, MACDStrategy, 
        BollingerBandsStrategy, MomentumBreakoutStrategy
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"模块导入失败: {e}")
    MODULES_AVAILABLE = False

def main():
    """主应用"""
    st.title("📈 Tony&Associates QuantAI Trader")
    st.markdown("*专业级量化交易系统 - 集成数据获取、策略回测、机器学习预测*")
    
    # 初始化会话状态
    init_session_state()
    
    # 侧边栏导航
    with st.sidebar:
        st.header("🎛️ 控制面板")
        menu = st.selectbox(
            "选择功能模块",
            ["🏠 系统概览", "🔍 选股工具", "📊 市场数据", "🎯 策略回测", "🤖 机器学习"]
        )
    
    # 主菜单路由
    if menu == "🏠 系统概览":
        show_system_overview()
    elif menu == "🔍 选股工具":
        show_stock_selector()
    elif menu == "📊 市场数据":
        show_data_interface()
    elif menu == "🎯 策略回测":
        show_backtest_interface()
    elif menu == "🤖 机器学习":
        show_ml_interface()

def init_session_state():
    """初始化会话状态"""
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None
    if 'backtest_stock' not in st.session_state:
        st.session_state.backtest_stock = None
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    if 'backtest_history' not in st.session_state:
        st.session_state.backtest_history = []

@st.cache_resource
def initialize_system():
    """初始化量化系统"""
    if not MODULES_AVAILABLE:
        return None
    try:
        system = create_quant_system()
        return system
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        return None

def get_market_info(symbol):
    """判断股票市场类型"""
    symbol = symbol.upper().strip()
    
    if symbol.isdigit() and len(symbol) == 6:
        if symbol.startswith(('00', '30')):
            return 'A', '深市'
        elif symbol.startswith('60'):
            return 'A', '沪市'
        else:
            return 'A', 'A股'
    elif '.' in symbol:
        if symbol.endswith('.HK'):
            return 'HK', '港股'
        elif symbol.endswith('.SS'):
            return 'A', '沪市'
        elif symbol.endswith('.SZ'):
            return 'A', '深市'
    else:
        return 'US', '美股'
    
    return 'US', '美股'

def show_system_overview():
    """系统概览"""
    st.subheader("🎯 系统功能概览")
    
    # 系统状态检查
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if MODULES_AVAILABLE:
            st.markdown('<div class="success-banner">✅ 系统正常</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-banner">❌ 系统异常</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h4>数据源</h4><p>Akshare + Tushare</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h4>支持市场</h4><p>A股 + 港股 + 美股</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h4>策略数量</h4><p>5+ 种策略</p></div>', unsafe_allow_html=True)
    
    # 快速开始
    st.subheader("⚡ 快速开始")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        symbol = st.text_input("股票代码", value="AAPL", help="支持代码/拼音/中文，如: AAPL, 000001, pingan")
        market, market_desc = get_market_info(symbol)
        st.markdown(f'<span class="market-badge market-{market.lower()}">{market_desc}</span>', unsafe_allow_html=True)
        
        strategy = st.selectbox("选择策略", [
            "moving_average", "rsi", "macd", "bollinger_bands", "momentum"
        ], format_func=lambda x: {
            "moving_average": "📈 移动平均策略",
            "rsi": "📊 RSI策略", 
            "macd": "🔄 MACD策略",
            "bollinger_bands": "📏 布林带策略",
            "momentum": "🚀 动量策略"
        }.get(x, x))
    
    with col2:
        days = st.slider("回测天数", 30, 365, 90)
        
        # 默认参数说明
        with st.expander("📋 默认参数设置"):
            st.markdown("""
            **一键回测默认参数:**
            - 初始资金: 100,000 元
            - 手续费率: 0.1%
            - 印花税: 0.1%
            - 滑点: 忽略
            - 风险控制: 启用
            
            **策略参数:**
            - 移动平均: 快线5日, 慢线20日
            - RSI: 周期14, 超买70, 超卖30
            - MACD: 快线12, 慢线26, 信号线9
            - 布林带: 周期20, 标准差2
            - 动量策略: 突破周期20
            """)
        
        if st.button("🚀 一键回测", type="primary", use_container_width=True):
            run_quick_backtest(symbol, strategy, days)

def run_quick_backtest(symbol, strategy, days):
    """执行快速回测"""
    with st.spinner("正在执行回测..."):
        try:
            if not MODULES_AVAILABLE:
                st.error("模块未正确加载，使用演示模式")
                show_demo_results(symbol, strategy)
                return
                
            system = initialize_system()
            if not system:
                st.error("系统未正确初始化，使用演示模式")
                show_demo_results(symbol, strategy)
                return
                
            # 获取数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            data_result = system.data_engine.get_data_pipeline(symbol, start_date, end_date)
            
            if data_result.get('status') != 'success':
                st.error(f"数据获取失败: {data_result.get('message', '未知错误')}")
                show_demo_results(symbol, strategy)
                return
            
            stock_data = {symbol: data_result['processed_data']}
            
            # 创建策略实例
            strategy_map = {
                "moving_average": MovingAverageCrossStrategy(fast=5, slow=20),
                "rsi": RSIStrategy(period=14, overbought=70, oversold=30),
                "macd": MACDStrategy(fast=12, slow=26, signal=9),
                "bollinger_bands": BollingerBandsStrategy(window=20, num_std=2),
                "momentum": MomentumBreakoutStrategy(window=20)
            }
            
            strategy_instance = strategy_map.get(strategy)
            if not strategy_instance:
                st.error(f"未知策略: {strategy}")
                return
            
            # 运行回测
            backtest_engine = BacktestEngine(commission=0.001, tax=0.001)
            
            def strategy_func(df):
                return strategy_instance.generate_signal(df)
            
            results = backtest_engine.run(
                stock_data=stock_data,
                strategy_func=strategy_func,
                initial_cash=100000
            )
            
            # 显示结果
            show_backtest_results(results, symbol, strategy)
            
        except Exception as e:
            st.error(f"执行失败: {e}")
            show_demo_results(symbol, strategy)

def show_demo_results(symbol, strategy):
    """显示演示结果"""
    st.info("🔄 演示模式：显示模拟回测结果")
    
    # 模拟结果
    total_return = np.random.uniform(-0.2, 0.5)
    annual_return = total_return * 365 / 90
    max_drawdown = np.random.uniform(0.05, 0.25)
    sharpe = np.random.uniform(0.5, 2.5)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总收益率", f"{total_return*100:.2f}%")
    with col2:
        st.metric("年化收益率", f"{annual_return*100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{max_drawdown*100:.2f}%")
    with col4:
        st.metric("夏普比率", f"{sharpe:.3f}")
    
    # 模拟资金曲线
    dates = pd.date_range(datetime.now() - timedelta(days=90), periods=90, freq='D')
    equity_curve = [100000]
    for i in range(89):
        daily_return = np.random.normal(total_return/90, 0.02)
        equity_curve.append(equity_curve[-1] * (1 + daily_return))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity_curve,
        mode='lines',
        name='策略收益',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=f"{symbol} - {strategy} 策略回测结果 (演示)",
        xaxis_title="日期",
        yaxis_title="资金 (元)",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.warning("⚠️ 这是演示数据，实际使用需要配置真实数据源")

def show_backtest_results(results, symbol, strategy):
    """显示回测结果"""
    if not results:
        st.error("回测结果为空")
        return
        
    st.success("✅ 回测完成!")
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = results.get('total_return', 0) * 100
        st.metric("总收益率", f"{total_return:.2f}%")
    
    with col2:
        annual_return = results.get('annual_return', 0) * 100
        st.metric("年化收益率", f"{annual_return:.2f}%")
    
    with col3:
        max_drawdown = results.get('max_drawdown', 0) * 100
        st.metric("最大回撤", f"{max_drawdown:.2f}%")
    
    with col4:
        sharpe = results.get('sharpe', 0)
        st.metric("夏普比率", f"{sharpe:.3f}")
    
    # 资金曲线图
    if 'equity_curve' in results and results['equity_curve'] is not None:
        st.subheader("📈 策略资金曲线")
        
        equity_data = results['equity_curve']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_data.index,
            y=equity_data.values,
            mode='lines',
            name='策略收益',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} - {strategy} 策略回测结果",
            xaxis_title="日期",
            yaxis_title="资金 (元)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_stock_selector():
    """选股工具"""
    st.subheader("🔍 智能选股工具")
    st.info("📊 A股实时选股功能")
    
    # 筛选条件
    col1, col2 = st.columns(2)
    
    with col1:
        pe_min = st.number_input("PE比率最小值", value=0.0, step=0.1)
        pe_max = st.number_input("PE比率最大值", value=50.0, step=0.1)
    
    with col2:
        price_min = st.number_input("价格最小值", value=0.0, step=0.1)
        price_max = st.number_input("价格最大值", value=1000.0, step=0.1)
    
    if st.button("🔍 开始筛选", type="primary"):
        run_stock_screening(pe_min, pe_max, price_min, price_max)

def run_stock_screening(pe_min, pe_max, price_min, price_max):
    """执行选股筛选"""
    with st.spinner("正在筛选股票..."):
        try:
            # 尝试使用真实数据
            import akshare as ak
            stock_list = ak.stock_zh_a_spot_em()
            stock_list = stock_list.head(50)  # 限制数量
            
            filtered_stocks = []
            for idx, row in stock_list.head(20).iterrows():
                try:
                    symbol = row['代码']
                    name = row['名称']
                    pe = row.get('市盈率-动态', 0)
                    price = row.get('最新价', 0)
                    
                    # 应用筛选条件
                    if (pe_min <= pe <= pe_max and 
                        price_min <= price <= price_max and
                        pe > 0 and price > 0):
                        
                        filtered_stocks.append({
                            'symbol': symbol,
                            'name': name,
                            'pe': pe,
                            'price': price,
                            'change_pct': row.get('涨跌幅', 0)
                        })
                except Exception:
                    continue
            
            if filtered_stocks:
                result_df = pd.DataFrame(filtered_stocks)
                st.success(f"筛选完成! 找到 {len(result_df)} 只股票")
                st.dataframe(result_df, use_container_width=True)
                
                # 添加导入功能
                st.subheader("📈 选股结果操作")
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_index = st.selectbox(
                        "选择股票进行分析", 
                        range(len(result_df)),
                        format_func=lambda x: f"{result_df.iloc[x]['symbol']} - {result_df.iloc[x]['name']}"
                    )
                
                with col2:
                    if st.button("📊 导入市场数据", type="primary"):
                        selected_stock = result_df.iloc[selected_index]['symbol']
                        st.session_state.selected_stock = selected_stock
                        st.success(f"✅ 已选择 {selected_stock}，可切换到市场数据模块查看")
                    
                    if st.button("🎯 导入策略回测", type="secondary"):
                        selected_stock = result_df.iloc[selected_index]['symbol']
                        st.session_state.backtest_stock = selected_stock
                        st.success(f"✅ 已选择 {selected_stock}，可切换到策略回测模块")
            else:
                st.warning("未找到符合条件的股票")
                
        except Exception as e:
            st.error(f"筛选失败: {e}")
            st.info("💡 选股功能需要配置akshare数据源")

def show_data_interface():
    """市场数据界面"""
    st.subheader("📊 市场数据获取")
    
    # 检查导入的股票
    default_symbol = "AAPL"
    if st.session_state.selected_stock:
        default_symbol = st.session_state.selected_stock
        st.info(f"✨ 已从选股工具导入: {default_symbol}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        symbol = st.text_input("股票代码", default_symbol, key="data_symbol")
        market, market_desc = get_market_info(symbol)
        st.markdown(f'<span class="market-badge market-{market.lower()}">{market_desc}</span>', 
                   unsafe_allow_html=True)
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    
    with col2:
        end_date = st.date_input("结束日期", datetime.now())
        st.markdown("**数据源**: Akshare + Tushare (真实数据)")
    
    if st.button("📊 获取数据", type="primary"):
        show_market_data_demo(symbol, start_date, end_date)

def show_market_data_demo(symbol, start_date, end_date):
    """显示市场数据演示"""
    st.info("📊 数据获取演示模式")
    
    # 生成模拟数据
    date_range = pd.date_range(start_date, end_date, freq='D')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.normal(0, 2, len(date_range))),
        'volume': np.random.randint(1000000, 10000000, len(date_range))
    }, index=date_range)
    
    # 数据信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("数据记录数", len(data))
    with col2:
        st.metric("数据来源", "演示数据")
    with col3:
        st.metric("时间跨度", f"{len(data)}天")
    
    # 价格图表
    st.subheader("📈 价格走势图")
    
    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.radio("图表类型", ["收盘价走势", "K线图"], horizontal=True)
    with col2:
        time_period = st.selectbox("时间周期", ["1分钟", "5分钟", "1小时", "1日", "周", "月"], index=3)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        mode='lines',
        name='收盘价',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=f"{symbol} 价格走势 - {time_period}",
        xaxis_title="日期",
        yaxis_title="价格",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_backtest_interface():
    """策略回测界面"""
    st.subheader("🎯 策略回测系统")
    
    # 检查导入的股票
    default_symbol = "AAPL"
    if st.session_state.backtest_stock:
        default_symbol = st.session_state.backtest_stock
        st.info(f"✨ 已从选股工具导入: {default_symbol}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 回测参数")
        
        symbol = st.text_input("股票代码", default_symbol, key="backtest_symbol")
        market, market_desc = get_market_info(symbol)
        st.markdown(f'<span class="market-badge market-{market.lower()}">{market_desc}</span>', 
                   unsafe_allow_html=True)
        
        # 交易类型选择
        trading_type = st.selectbox("交易类型", [
            "日内交易", "趋势交易(中短期)", "价值投资(中长期)"
        ])
        
        # 根据交易类型限制时间周期
        if trading_type == "日内交易":
            if market == 'A':
                st.warning("⚠️ A股市场不支持日内交易，请选择其他交易类型")
                time_options = ["1日"]
            else:
                time_options = ["1分钟", "5分钟", "15分钟", "1小时"]
        elif trading_type == "趋势交易(中短期)":
            time_options = ["1分钟", "5分钟", "15分钟", "1小时", "1日", "周"]
        else:  # 价值投资
            time_options = ["1小时", "1日", "周", "月"]
        
        time_frame = st.selectbox("时间周期", time_options)
        
        strategy = st.selectbox("选择策略", [
            "moving_average", "rsi", "macd", "bollinger_bands", "momentum"
        ], format_func=lambda x: {
            "moving_average": "📈 移动平均策略",
            "rsi": "📊 RSI策略", 
            "macd": "🔄 MACD策略",
            "bollinger_bands": "📏 布林带策略",
            "momentum": "🚀 动量策略"
        }.get(x, x))
        
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
        end_date = st.date_input("结束日期", datetime.now())
        initial_capital = st.number_input("初始资金", min_value=10000, value=100000, step=10000)
        
        if st.button("🚀 运行回测", type="primary", use_container_width=True):
            run_strategy_backtest(symbol, strategy, start_date, end_date, initial_capital, trading_type, time_frame)
    
    with col2:
        st.markdown("#### 📊 回测结果")
        show_backtest_history()

def run_strategy_backtest(symbol, strategy, start_date, end_date, initial_capital, trading_type, time_frame):
    """运行策略回测"""
    with st.spinner("正在执行策略回测..."):
        try:
            if not MODULES_AVAILABLE:
                show_demo_backtest_results(symbol, strategy, initial_capital)
                return
                
            # 尝试真实回测
            system = initialize_system()
            if system:
                # 这里可以添加真实的回测逻辑
                show_demo_backtest_results(symbol, strategy, initial_capital)
            else:
                show_demo_backtest_results(symbol, strategy, initial_capital)
                
        except Exception as e:
            st.error(f"❌ 回测执行失败: {e}")
            show_demo_backtest_results(symbol, strategy, initial_capital)

def show_demo_backtest_results(symbol, strategy, initial_capital):
    """显示演示回测结果"""
    st.info("🔄 演示模式：显示模拟回测结果")
    
    # 模拟结果
    total_return = np.random.uniform(-0.2, 0.5)
    annual_return = total_return * 2
    max_drawdown = np.random.uniform(0.05, 0.25)
    sharpe = np.random.uniform(0.5, 2.5)
    win_rate = np.random.uniform(0.3, 0.8)
    
    # 保存到历史记录
    backtest_record = {
        'timestamp': datetime.now(),
        'symbol': symbol,
        'strategy': strategy,
        'total_return': total_return,
        'sharpe': sharpe
    }
    st.session_state.backtest_history.append(backtest_record)
    
    # 只保留最近5次记录
    if len(st.session_state.backtest_history) > 5:
        st.session_state.backtest_history = st.session_state.backtest_history[-5:]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总收益率", f"{total_return*100:.2f}%")
    with col2:
        st.metric("年化收益率", f"{annual_return*100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{max_drawdown*100:.2f}%")
    with col4:
        st.metric("夏普比率", f"{sharpe:.3f}")
    
    # 更多指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("胜率", f"{win_rate*100:.1f}%")
    with col2:
        st.metric("平均持仓天数", f"{np.random.randint(3, 30)}")
    with col3:
        st.metric("盈亏比", f"{np.random.uniform(1.0, 3.0):.2f}")
    with col4:
        st.metric("索提诺比率", f"{np.random.uniform(0.5, 2.0):.3f}")
    
    # 模拟股票走势 + 买卖点
    st.subheader("📈 股票走势与交易信号")
    
    dates = pd.date_range(datetime.now() - timedelta(days=90), periods=90, freq='D')
    prices = [100]
    for i in range(89):
        daily_return = np.random.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + daily_return))
    
    # 模拟买卖点
    buy_dates = np.random.choice(dates, size=5, replace=False)
    sell_dates = np.random.choice(dates, size=5, replace=False)
    
    fig = go.Figure()
    
    # 股价走势
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='股价',
        line=dict(color='blue', width=2)
    ))
    
    # 买入点
    fig.add_trace(go.Scatter(
        x=buy_dates,
        y=[prices[list(dates).index(d)] for d in buy_dates],
        mode='markers',
        name='买入点',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))
    
    # 卖出点
    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=[prices[list(dates).index(d)] for d in sell_dates],
        mode='markers',
        name='卖出点',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))
    
    fig.update_layout(
        title=f"{symbol} - {strategy} 策略交易信号 (演示)",
        xaxis_title="日期",
        yaxis_title="价格",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 交易记录
    st.subheader("📋 详细交易记录")
    
    trades_data = []
    for i, (buy_date, sell_date) in enumerate(zip(buy_dates[:3], sell_dates[:3])):
        buy_price = prices[list(dates).index(buy_date)]
        sell_price = prices[list(dates).index(sell_date)]
        qty = 100
        profit = (sell_price - buy_price) * qty
        
        trades_data.extend([
            {
                'date': buy_date,
                'type': 'BUY',
                'price': buy_price,
                'qty': qty,
                'cash': initial_capital - buy_price * qty,
                'profit': 0
            },
            {
                'date': sell_date,
                'type': 'SELL',
                'price': sell_price,
                'qty': qty,
                'cash': initial_capital - buy_price * qty + sell_price * qty,
                'profit': profit
            }
        ])
    
    trades_df = pd.DataFrame(trades_data)
    st.dataframe(trades_df, use_container_width=True)

def show_backtest_history():
    """显示回测历史记录"""
    if st.session_state.backtest_history:
        st.subheader("📚 最近回测记录")
        
        for i, record in enumerate(reversed(st.session_state.backtest_history)):
            with st.expander(f"回测 {i+1}: {record['symbol']} - {record['strategy']} ({record['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**股票**: {record['symbol']}")
                    st.write(f"**策略**: {record['strategy']}")
                
                with col2:
                    total_return = record.get('total_return', 0) * 100
                    st.write(f"**总收益**: {total_return:.2f}%")
                    st.write(f"**夏普比率**: {record.get('sharpe', 0):.3f}")

def show_ml_interface():
    """机器学习界面"""
    st.subheader("🤖 机器学习预测")
    st.info("🚧 机器学习模块正在开发中，敬请期待...")
    
    # 简单的演示界面
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("股票代码", "AAPL", key="ml_symbol")
        model_type = st.selectbox("模型类型", ["随机森林", "LSTM", "XGBoost"])
        predict_days = st.slider("预测天数", 1, 30, 5)
    
    with col2:
        if st.button("🔮 开始预测", type="primary"):
            st.info("演示模式：显示模拟预测结果")
            
            # 模拟预测结果
            dates = pd.date_range(datetime.now().date(), periods=predict_days, freq='D')
            current_price = 150.0
            predicted_prices = [current_price * (1 + np.random.normal(0, 0.02)) for _ in range(predict_days)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("当前价格", f"${current_price:.2f}")
            with col2:
                final_price = predicted_prices[-1]
                st.metric("预测价格", f"${final_price:.2f}")
            with col3:
                change_pct = (final_price - current_price) / current_price * 100
                st.metric("预期涨跌", f"{change_pct:+.2f}%")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=predicted_prices,
                mode='lines+markers',
                name='预测价格',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{symbol} 价格预测 ({predict_days}天)",
                xaxis_title="日期",
                yaxis_title="价格",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 