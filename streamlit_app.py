#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tony&Associates QuantAI Trader - Streamlit主应用
量化交易系统Web界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
import os
import logging

# 导入集成系统
try:
    from integration import QuantSystem, create_quant_system, quick_backtest
    from strategy_core.strategy_core import (
        MovingAverageCrossStrategy, RSIStrategy, MACDStrategy, 
        BollingerBandsStrategy, MomentumBreakoutStrategy
    )
    from strategy_core.stock_selector import StockSelector
    from strategy_core.backtest_engine import BacktestEngine
except ImportError as e:
    st.error(f"无法导入集成系统模块: {e}")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="Tony&Associates QuantAI Trader",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-banner {
        background: linear-gradient(90deg, #00C851 0%, #007E33 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .warning-banner {
        background: linear-gradient(90deg, #ffbb33 0%, #ff8800 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .error-banner {
        background: linear-gradient(90deg, #ff4444 0%, #cc0000 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .market-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .market-a { background: #ff6b6b; color: white; }
    .market-hk { background: #4ecdc4; color: white; }
    .market-us { background: #45b7d1; color: white; }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-card {
            margin: 0.2rem 0;
            padding: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Session State初始化
if 'system' not in st.session_state:
    st.session_state.system = None
if 'results' not in st.session_state:
    st.session_state.results = None

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 主标题
    st.markdown('<h1 class="main-header">🚀 Tony&Associates QuantAI Trader</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI驱动的量化交易系统 - 真实数据模式</p>', unsafe_allow_html=True)
    
    # 侧边栏导航
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>📊 控制面板</h2>
        </div>
        """, unsafe_allow_html=True)
        
        menu = st.selectbox(
            "选择功能模块",
            ["🏠 系统概览", "🔍 选股工具", "📈 市场数据", "🎯 策略回测", "🤖 机器学习", "🚀 部署指南"]
        )
    
    # 主菜单路由
    if menu == "🏠 系统概览":
        show_system_overview()
    elif menu == "🔍 选股工具":
        show_stock_selector()
    elif menu == "📈 市场数据":
        show_data_interface()
    elif menu == "🎯 策略回测":
        show_backtest_interface()
    elif menu == "🤖 机器学习":
        show_ml_interface()
    elif menu == "🚀 部署指南":
        show_deployment_guide()

@st.cache_resource
def initialize_system():
    """初始化量化系统"""
    try:
        system = create_quant_system()
        return system
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        logger.error(f"System initialization failed: {e}")
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
        # 美股或其他
        return 'US', '美股'
    
    return 'US', '美股'

def show_system_overview():
    """显示系统概览"""
    st.subheader("🎯 系统功能概览")
    
    # 系统状态检查
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        system = initialize_system()
        if system:
            st.markdown('<div class="success-banner">✅ 系统正常</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-banner">❌ 系统异常</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h4>数据源</h4><p>Akshare + Tushare</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h4>支持市场</h4><p>A股 + 港股 + 美股</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h4>策略数量</h4><p>5+ 种策略</p></div>', unsafe_allow_html=True)
    
    # 功能卡片
    st.subheader("📚 功能模块")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 智能选股</h3>
            <p>多市场股票筛选</p>
            <ul style="text-align: left;">
                <li>A股/港股/美股</li>
                <li>代码/拼音识别</li>
                <li>规则筛选</li>
                <li>ML智能选股</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 数据引擎</h3>
            <p>真实数据获取处理</p>
            <ul style="text-align: left;">
                <li>实时股票数据</li>
                <li>技术指标计算</li>
                <li>特征工程</li>
                <li>数据一致性检查</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 策略引擎</h3>
            <p>专业策略回测系统</p>
            <ul style="text-align: left;">
                <li>移动平均/RSI/MACD</li>
                <li>布林带/动量策略</li>
                <li>参数优化</li>
                <li>风险管理</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速开始
    st.subheader("⚡ 快速开始")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        symbol = st.text_input("股票代码", value="AAPL", help="支持代码/拼音/中文，如: AAPL, 000001, pingan")
        market, market_desc = get_market_info(symbol)
        st.markdown(f'<span class="market-badge market-{market.lower()}">{market_desc}</span>', unsafe_allow_html=True)
        
        strategy = st.selectbox("选择策略", ["moving_average", "rsi", "macd", "bollinger_bands"])
    
    with col2:
        days = st.slider("回测天数", 30, 365, 90)
        
        if st.button("🚀 一键回测", type="primary", use_container_width=True):
            with st.spinner("正在执行回测..."):
                try:
                    system = initialize_system()
                    if system:
                        # 获取数据
                        end_date = datetime.now().strftime('%Y%m%d')
                        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
                        
                        results = system.run_full_pipeline(
                            symbol=symbol,
                            strategy_name=strategy,
                            start_date=start_date,
                            end_date=end_date,
                            initial_capital=100000
                        )
                        
                        if results.get('status') == 'success':
                            show_quick_results(results)
                        else:
                            st.error(f"回测失败: {results.get('message', '未知错误')}")
                    else:
                        st.error("系统未正确初始化")
                except Exception as e:
                    st.error(f"执行失败: {e}")

def show_quick_results(results):
    """显示快速回测结果"""
    if 'strategy' in results and results['strategy'].get('status') == 'success':
        backtest = results['strategy']['backtest_results']
        
        st.success("✅ 回测完成!")
        
        # 关键指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = backtest.get('total_return', 0) * 100
            st.metric("总收益率", f"{total_return:.2f}%")
        with col2:
            sharpe = backtest.get('sharpe', 0)
            st.metric("夏普比率", f"{sharpe:.3f}")
        with col3:
            max_dd = backtest.get('max_drawdown', 0) * 100
            st.metric("最大回撤", f"{max_dd:.2f}%")
        with col4:
            win_rate = backtest.get('win_rate', 0) * 100
            st.metric("胜率", f"{win_rate:.1f}%")

def show_stock_selector():
    """选股工具界面"""
    st.subheader("🔍 智能选股工具")
    
    tab1, tab2 = st.tabs(["📋 规则选股", "🤖 AI选股"])
    
    with tab1:
        st.markdown("#### 基于规则的股票筛选")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            market = st.selectbox("选择市场", ["A", "HK", "US"], format_func=lambda x: {
                "A": "🇨🇳 A股市场", 
                "HK": "🇭🇰 港股市场", 
                "US": "🇺🇸 美股市场"
            }[x])
            
            # 规则配置
            rules = []
            st.markdown("**筛选条件:**")
            
            if st.checkbox("市盈率筛选"):
                pe_min = st.number_input("PE最小值", value=0.0)
                pe_max = st.number_input("PE最大值", value=50.0)
                rules.append(f"pe>={pe_min}")
                rules.append(f"pe<={pe_max}")
            
            if st.checkbox("成交量筛选"):
                vol_ratio = st.number_input("成交量比率", value=1.5, help="相对平均成交量的倍数")
                rules.append(f"volume_ratio>{vol_ratio}")
            
            if st.checkbox("价格突破"):
                ma_period = st.selectbox("均线周期", [5, 10, 20, 60])
                rules.append(f"Close>MA{ma_period}")
        
        with col2:
            if st.button("🔍 开始筛选", type="primary"):
                if rules:
                    with st.spinner("正在筛选股票..."):
                        try:
                            system = initialize_system()
                            if system:
                                # 这里应该调用实际的选股功能
                                # 由于没有现成的股票列表数据，我们模拟一个结果
                                st.success("筛选完成!")
                                
                                # 模拟筛选结果
                                sample_stocks = pd.DataFrame({
                                    'symbol': ['000001', '000002', '600519'],
                                    'name': ['平安银行', '万科A', '贵州茅台'],
                                    'pe': [8.5, 12.3, 45.2],
                                    'volume_ratio': [1.8, 2.1, 1.6],
                                    'price': [12.45, 23.67, 1890.50]
                                })
                                
                                st.dataframe(sample_stocks, use_container_width=True)
                                st.info("注: 这是模拟数据，实际选股功能需要连接到实时数据源")
                        except Exception as e:
                            st.error(f"筛选失败: {e}")
                else:
                    st.warning("请设置至少一个筛选条件")
    
    with tab2:
        st.markdown("#### AI智能选股")
        st.info("🚧 AI选股功能开发中...")

def show_data_interface():
    """数据引擎界面"""
    st.subheader("📊 数据引擎")
    
    tab1, tab2, tab3 = st.tabs(["📈 数据获取", "🔧 数据处理", "🎯 特征工程"])
    
    with tab1:
        st.markdown("#### 股票数据获取")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            symbol = st.text_input("股票代码", "AAPL", key="data_symbol", 
                                 help="支持多种格式: AAPL, 000001, 00700.HK, 平安银行")
            
            market, market_desc = get_market_info(symbol)
            st.markdown(f'<span class="market-badge market-{market.lower()}">{market_desc}</span>', 
                       unsafe_allow_html=True)
            
            start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
            
        with col2:
            end_date = st.date_input("结束日期", datetime.now())
            
            # 移除不支持的数据源
            st.markdown("**数据源**: Akshare + Tushare (真实数据)")
            st.info("💡 系统自动选择最佳数据源")
        
        if st.button("📊 获取数据", type="primary"):
            with st.spinner("正在获取真实股票数据..."):
                try:
                    system = initialize_system()
                    if system:
                        # 调用真实的数据获取接口
                        data_result = system.data_engine.get_data_pipeline(
                            symbol, 
                            start_date.strftime('%Y%m%d'), 
                            end_date.strftime('%Y%m%d')
                        )
                        
                        if data_result.get('status') == 'success':
                            st.success("✅ 数据获取成功!")
                            
                            # 显示数据信息
                            data = data_result['processed_data']
                            features = data_result.get('features', pd.DataFrame())
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("数据记录数", len(data))
                            with col2:
                                st.metric("数据来源", data_result.get('source', 'unknown'))
                            with col3:
                                st.metric("特征数量", len(features.columns) if not features.empty else 0)
                            
                            # 显示最新数据
                            st.subheader("📋 最新数据")
                            st.dataframe(data.tail(10), use_container_width=True)
                            
                            # 价格图表
                            if 'close' in data.columns:
                                st.subheader("📈 价格走势图")
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=data['close'],
                                    mode='lines',
                                    name='收盘价',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                fig.update_layout(
                                    title=f"{symbol} 价格走势 ({market_desc})",
                                    xaxis_title="日期",
                                    yaxis_title="价格",
                                    height=400,
                                    template="plotly_white"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # 技术指标图表
                            if not features.empty and 'ma_5' in features.columns:
                                st.subheader("📊 技术指标")
                                fig2 = go.Figure()
                                
                                if 'ma_5' in features.columns:
                                    fig2.add_trace(go.Scatter(
                                        x=features.index,
                                        y=features['ma_5'],
                                        name='MA5',
                                        line=dict(color='orange')
                                    ))
                                
                                if 'ma_20' in features.columns:
                                    fig2.add_trace(go.Scatter(
                                        x=features.index,
                                        y=features['ma_20'],
                                        name='MA20',
                                        line=dict(color='red')
                                    ))
                                
                                fig2.update_layout(
                                    title="移动平均线",
                                    xaxis_title="日期",
                                    yaxis_title="价格",
                                    height=300,
                                    template="plotly_white"
                                )
                                
                                st.plotly_chart(fig2, use_container_width=True)
                        else:
                            error_msg = data_result.get('message', '未知错误')
                            st.error(f"❌ 数据获取失败: {error_msg}")
                            
                            # 给出解决建议
                            if "akshare" in error_msg.lower():
                                st.info("💡 建议: 检查网络连接或稍后重试")
                            elif "symbol" in error_msg.lower():
                                st.info("💡 建议: 检查股票代码格式是否正确")
                except Exception as e:
                    st.error(f"❌ 系统错误: {e}")
                    logger.error(f"Data fetching error: {e}")
    
    with tab2:
        st.markdown("#### 数据预处理功能")
        
        processing_options = st.multiselect(
            "选择处理选项",
            ["缺失值处理", "异常值检测", "数据标准化", "时间序列对齐"],
            help="选择需要应用的数据预处理步骤"
        )
        
        if st.button("🔧 执行处理"):
            if processing_options:
                st.success("✅ 数据处理完成!")
                st.info("注: 当前显示模拟结果，实际处理逻辑已集成在数据获取流程中")
            else:
                st.warning("请选择至少一个处理选项")
    
    with tab3:
        st.markdown("#### 技术指标与特征工程")
        
        feature_types = st.multiselect(
            "选择特征类型",
            ["移动平均线", "RSI指标", "MACD指标", "布林带", "成交量指标"],
            help="选择要生成的技术指标"
        )
        
        if st.button("⚙️ 生成特征"):
            if feature_types:
                st.success("✅ 特征生成完成!")
                st.info("注: 技术指标会在数据获取时自动计算")
            else:
                st.warning("请选择至少一个特征类型")

def show_backtest_interface():
    """策略回测界面"""
    st.subheader("🎯 策略回测系统")
    
    # 参数设置
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 回测参数")
        
        # 股票选择
        symbol = st.text_input("股票代码", "AAPL", key="backtest_symbol",
                              help="支持多种格式: AAPL, 000001, 00700.HK")
        market, market_desc = get_market_info(symbol)
        st.markdown(f'<span class="market-badge market-{market.lower()}">{market_desc}</span>', 
                   unsafe_allow_html=True)
        
        # 策略选择
        strategy = st.selectbox("选择策略", [
            "moving_average", "rsi", "macd", "bollinger_bands", "momentum"
        ], format_func=lambda x: {
            "moving_average": "📈 移动平均策略",
            "rsi": "📊 RSI策略", 
            "macd": "🔄 MACD策略",
            "bollinger_bands": "📏 布林带策略",
            "momentum": "🚀 动量策略"
        }.get(x, x))
        
        # 时间设置
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365), key="backtest_start")
        end_date = st.date_input("结束日期", datetime.now(), key="backtest_end")
        
        # 资金设置
        initial_capital = st.number_input("初始资金", min_value=10000, value=100000, step=10000,
                                        help="回测初始资金额度")
        
        # 高级选项
        with st.expander("🔧 高级设置"):
            optimize = st.checkbox("参数优化", help="自动优化策略参数")
            use_ml = st.checkbox("机器学习辅助", help="使用ML模型辅助决策")
            
            commission = st.number_input("手续费率", value=0.001, format="%.4f", help="交易手续费率")
            slippage = st.number_input("滑点", value=0.001, format="%.4f", help="交易滑点")
        
        run_backtest = st.button("🚀 运行回测", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 回测结果")
        
        if run_backtest:
            with st.spinner("正在执行策略回测..."):
                try:
                    system = initialize_system()
                    if system:
                        # 执行完整的回测流程
                        results = system.run_full_pipeline(
                            symbol=symbol,
                            strategy_name=strategy,
                            start_date=start_date.strftime('%Y%m%d'),
                            end_date=end_date.strftime('%Y%m%d'),
                            initial_capital=initial_capital,
                            use_ml=use_ml,
                            optimize_params=optimize
                        )
                        
                        st.session_state.results = results
                        
                        if results.get('status') == 'success':
                            show_backtest_results(results)
                        else:
                            st.error(f"❌ 回测失败: {results.get('message', '未知错误')}")
                            
                except Exception as e:
                    st.error(f"❌ 回测执行失败: {e}")
                    logger.error(f"Backtest error: {e}")
        
        elif st.session_state.results:
            show_backtest_results(st.session_state.results)
        else:
            st.info("💡 配置参数后点击运行回测")
            
            # 显示策略说明
            strategy_info = {
                "moving_average": "基于快慢均线交叉的趋势跟踪策略",
                "rsi": "基于RSI指标的超买超卖策略", 
                "macd": "基于MACD指标的动量策略",
                "bollinger_bands": "基于布林带的均值回归策略",
                "momentum": "基于价格动量的突破策略"
            }
            
            if strategy in strategy_info:
                st.info(f"📝 策略说明: {strategy_info[strategy]}")

def show_backtest_results(results):
    """显示详细回测结果"""
    if 'strategy' in results and results['strategy'].get('status') == 'success':
        backtest = results['strategy']['backtest_results']
        
        st.success("✅ 回测完成!")
        
        # 关键指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = backtest.get('total_return', 0) * 100
            delta_color = "normal" if total_return >= 0 else "inverse"
            st.metric("总收益率", f"{total_return:.2f}%", delta_color=delta_color)
            
        with col2:
            sharpe = backtest.get('sharpe', 0)
            st.metric("夏普比率", f"{sharpe:.3f}")
            
        with col3:
            max_dd = backtest.get('max_drawdown', 0) * 100
            st.metric("最大回撤", f"{max_dd:.2f}%")
            
        with col4:
            win_rate = backtest.get('win_rate', 0) * 100
            st.metric("胜率", f"{win_rate:.1f}%")
        
        # 更多指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            annual_return = backtest.get('annual_return', 0) * 100
            st.metric("年化收益率", f"{annual_return:.2f}%")
            
        with col2:
            sortino = backtest.get('sortino', 0)
            st.metric("索提诺比率", f"{sortino:.3f}")
            
        with col3:
            avg_holding = backtest.get('avg_holding_days', 0)
            st.metric("平均持仓天数", f"{avg_holding:.1f}")
            
        with col4:
            profit_loss_ratio = backtest.get('profit_loss_ratio', 0)
            st.metric("盈亏比", f"{profit_loss_ratio:.2f}")
        
        # 资金曲线图
        if 'equity_curve' in backtest and backtest['equity_curve'] is not None:
            st.subheader("📈 策略资金曲线")
            
            equity_data = backtest['equity_curve']
            if hasattr(equity_data, 'values'):
                equity_values = equity_data.values
                equity_index = equity_data.index
            else:
                equity_values = equity_data
                equity_index = range(len(equity_values))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_index,
                y=equity_values,
                mode='lines',
                name='策略收益',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="策略资金曲线",
                xaxis_title="交易日",
                yaxis_title="资金 (元)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 交易记录
        if 'trades' in backtest and not backtest['trades'].empty:
            st.subheader("📋 交易记录")
            trades_df = backtest['trades']
            st.dataframe(trades_df.tail(10), use_container_width=True)
            
            # 下载交易记录
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 下载完整交易记录",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # 投资建议
    if 'analysis' in results and 'recommendations' in results['analysis']:
        st.subheader("💡 AI投资建议")
        analysis = results['analysis']
        for i, rec in enumerate(analysis['recommendations'], 1):
            st.write(f"{i}. {rec}")

def show_ml_interface():
    """机器学习界面"""
    st.subheader("🤖 机器学习预测")
    
    tab1, tab2 = st.tabs(["🎯 价格预测", "📊 模型分析"])
    
    with tab1:
        st.markdown("#### AI价格预测模型")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            symbol = st.text_input("股票代码", "AAPL", key="ml_symbol")
            market, market_desc = get_market_info(symbol)
            st.markdown(f'<span class="market-badge market-{market.lower()}">{market_desc}</span>', 
                       unsafe_allow_html=True)
            
            model_type = st.selectbox("模型类型", [
                "随机森林", "XGBoost", "LSTM", "线性回归"
            ])
            
        with col2:
            target_days = st.slider("预测天数", 1, 30, 5)
            confidence = st.slider("置信度", 0.8, 0.99, 0.95)
            
        if st.button("🔮 开始预测", type="primary"):
            with st.spinner("正在训练模型并预测..."):
                # 模拟ML预测结果
                st.success("✅ 预测完成!")
                
                # 模拟预测数据
                current_price = 150.0
                predicted_prices = [current_price * (1 + np.random.normal(0, 0.02)) for _ in range(target_days)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("当前价格", f"${current_price:.2f}")
                with col2:
                    final_price = predicted_prices[-1]
                    st.metric("预测价格", f"${final_price:.2f}")
                with col3:
                    change_pct = (final_price - current_price) / current_price * 100
                    st.metric("预期涨跌", f"{change_pct:+.2f}%")
                
                # 预测图表
                fig = go.Figure()
                dates = [datetime.now() + timedelta(days=i) for i in range(target_days)]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=predicted_prices,
                    mode='lines+markers',
                    name='价格预测',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} 价格预测 ({target_days}天)",
                    xaxis_title="日期",
                    yaxis_title="价格",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("⚠️ 这是演示功能，实际预测需要更多历史数据和模型训练")
    
    with tab2:
        st.markdown("#### 模型性能分析")
        st.info("🚧 模型分析功能开发中...")

def show_deployment_guide():
    """部署指南"""
    st.subheader("🚀 部署指南")
    
    tab1, tab2, tab3 = st.tabs(["☁️ 云端部署", "🐳 Docker部署", "📊 监控运维"])
    
    with tab1:
        st.markdown("""
        ### Streamlit Cloud部署
        
        1. **推送代码到GitHub**
        ```bash
        git add .
        git commit -m "Update quantai trader"
        git push origin main
        ```
        
        2. **配置Streamlit Cloud**
        - 访问 [share.streamlit.io](https://share.streamlit.io)
        - 连接GitHub仓库
        - 选择主分支和streamlit_app.py
        
        3. **环境配置**
        - requirements.txt: 已配置Python 3.11
        - runtime.txt: 指定Python版本
        """)
    
    with tab2:
        st.markdown("""
        ### Docker容器化部署
        
        ```dockerfile
        FROM python:3.11-slim
        
        WORKDIR /app
        COPY . .
        
        RUN pip install -r requirements.txt
        
        EXPOSE 8501
        
        CMD ["streamlit", "run", "streamlit_app.py"]
        ```
        """)
    
    with tab3:
        st.markdown("""
        ### 系统监控
        
        - **性能监控**: CPU、内存使用率
        - **数据监控**: 数据获取成功率
        - **策略监控**: 回测性能指标
        - **错误监控**: 异常日志记录
        """)

if __name__ == "__main__":
    main() 