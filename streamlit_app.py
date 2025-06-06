#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuantAI Trader - Streamlit主应用
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

# 导入集成系统
try:
    from integration import QuantSystem, create_quant_system, quick_backtest
except ImportError:
    st.error("无法导入集成系统模块，请检查文件路径")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="QuantAI Trader",
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
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
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

def main():
    """主函数"""
    # 主标题
    st.markdown('<h1 class="main-header">🚀 QuantAI Trader</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI驱动的量化交易系统</p>', unsafe_allow_html=True)
    
    # 侧边栏导航
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>📊 控制面板</h2>
        </div>
        """, unsafe_allow_html=True)
        
        menu = st.selectbox(
            "选择功能模块",
            ["🏠 系统概览", "📈 市场数据", "🎯 策略回测", "🤖 机器学习", "🚀 部署指南"]
        )
    
    # 主菜单路由
    if menu == "🏠 系统概览":
        show_system_overview()
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
        return create_quant_system()
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        return None

def show_system_overview():
    """显示系统概览"""
    st.subheader("🎯 系统功能概览")
    
    # 功能卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 数据引擎</h3>
            <p>多源数据获取与处理</p>
            <ul style="text-align: left;">
                <li>实时股票数据</li>
                <li>技术指标计算</li>
                <li>特征工程</li>
                <li>数据缓存</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 策略引擎</h3>
            <p>专业策略回测系统</p>
            <ul style="text-align: left;">
                <li>多种交易策略</li>
                <li>参数优化</li>
                <li>风险管理</li>
                <li>回测分析</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI引擎</h3>
            <p>机器学习价格预测</p>
            <ul style="text-align: left;">
                <li>多模型支持</li>
                <li>特征选择</li>
                <li>模型评估</li>
                <li>投资建议</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速开始
    st.subheader("⚡ 快速开始")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        symbol = st.text_input("股票代码", value="AAPL", help="输入股票代码，如 AAPL, TSLA, MSFT")
        strategy = st.selectbox("选择策略", ["moving_average", "rsi", "macd"])
    
    with col2:
        days = st.slider("回测天数", 30, 365, 90)
        
        if st.button("🚀 一键回测", type="primary", use_container_width=True):
            with st.spinner("正在执行回测..."):
                try:
                    results = quick_backtest(symbol, strategy, days)
                    st.session_state.results = results
                    
                    if results.get('strategy', {}).get('status') == 'success':
                        st.success("回测完成！")
                        show_quick_results(results)
                    else:
                        st.error(f"回测失败: {results.get('message', '未知错误')}")
                        
                except Exception as e:
                    st.error(f"执行失败: {e}")

def show_quick_results(results):
    """显示快速回测结果"""
    if 'strategy' in results and results['strategy']['status'] == 'success':
        backtest = results['strategy']['backtest_results']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总收益率", f"{backtest['total_return']:.2f}%")
        with col2:
            st.metric("夏普比率", f"{backtest['sharpe_ratio']:.2f}")
        with col3:
            st.metric("最大回撤", f"{backtest['max_drawdown']:.2f}%")
        with col4:
            st.metric("胜率", f"{backtest['win_rate']:.1f}%")
        
        # 资金曲线图
        if 'equity_curve' in backtest:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=backtest['equity_curve'],
                mode='lines',
                name='资金曲线',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="策略资金曲线",
                xaxis_title="时间",
                yaxis_title="资金 (元)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_data_interface():
    """数据引擎界面"""
    st.subheader("📊 数据引擎")
    
    tab1, tab2, tab3 = st.tabs(["📈 数据获取", "🔧 数据处理", "🎯 特征工程"])
    
    with tab1:
        st.markdown("#### 股票数据获取")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            symbol = st.text_input("股票代码", "AAPL", key="data_symbol")
            start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
            
        with col2:
            end_date = st.date_input("结束日期", datetime.now())
            data_source = st.selectbox("数据源", ["Yahoo Finance", "Alpha Vantage", "Tushare"])
        
        if st.button("获取数据", type="primary"):
            with st.spinner("正在获取数据..."):
                system = initialize_system()
                if system:
                    data_result = system.data_engine.get_data_pipeline(
                        symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                    )
                    
                    if data_result['status'] == 'success':
                        st.success("数据获取成功!")
                        
                        # 显示数据
                        data = data_result['processed_data']
                        st.dataframe(data.tail(10), use_container_width=True)
                        
                        # 价格图表
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['close'],
                            mode='lines',
                            name='收盘价',
                            line=dict(color='#1f77b4')
                        ))
                        
                        fig.update_layout(
                            title=f"{symbol} 价格走势",
                            xaxis_title="日期",
                            yaxis_title="价格",
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"数据获取失败: {data_result.get('message', '未知错误')}")
    
    with tab2:
        st.markdown("#### 数据预处理")
        
        processing_options = st.multiselect(
            "选择处理选项",
            ["缺失值处理", "异常值检测", "数据标准化", "时间序列对齐"]
        )
        
        if st.button("执行处理"):
            st.info("数据处理功能开发中...")
    
    with tab3:
        st.markdown("#### 技术指标与特征")
        
        feature_types = st.multiselect(
            "选择特征类型",
            ["技术指标", "价格特征", "成交量特征", "时间序列特征"]
        )
        
        if st.button("生成特征"):
            st.info("特征工程功能开发中...")

def show_backtest_interface():
    """策略回测界面"""
    st.subheader("🎯 策略回测系统")
    
    # 参数设置
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 回测参数")
        
        symbol = st.text_input("股票代码", "AAPL", key="backtest_symbol")
        strategy = st.selectbox("选择策略", [
            "moving_average", "rsi", "macd", "bollinger_bands", "momentum"
        ])
        
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365), key="backtest_start")
        end_date = st.date_input("结束日期", datetime.now(), key="backtest_end")
        
        initial_capital = st.number_input("初始资金", min_value=10000, value=1000000, step=10000)
        
        optimize = st.checkbox("参数优化", help="自动优化策略参数")
        use_ml = st.checkbox("机器学习辅助", help="使用ML模型辅助决策")
        
        run_backtest = st.button("🚀 运行回测", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 回测结果")
        
        if run_backtest:
            with st.spinner("正在执行回测..."):
                try:
                    system = initialize_system()
                    if system:
                        results = system.run_full_pipeline(
                            symbol=symbol,
                            strategy_name=strategy,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            initial_capital=initial_capital,
                            use_ml=use_ml,
                            optimize_params=optimize
                        )
                        
                        st.session_state.results = results
                        
                        if results.get('strategy', {}).get('status') == 'success':
                            show_backtest_results(results)
                        else:
                            st.error("回测失败，请检查参数设置")
                            
                except Exception as e:
                    st.error(f"回测执行失败: {e}")
        
        elif st.session_state.results:
            show_backtest_results(st.session_state.results)
        else:
            st.info("配置参数后点击运行回测")

def show_backtest_results(results):
    """显示回测结果"""
    if 'strategy' in results and results['strategy']['status'] == 'success':
        backtest = results['strategy']['backtest_results']
        
        # 关键指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总收益率", f"{backtest['total_return']:.2f}%")
        with col2:
            st.metric("夏普比率", f"{backtest['sharpe_ratio']:.2f}")
        with col3:
            st.metric("最大回撤", f"{backtest['max_drawdown']:.2f}%")
        with col4:
            st.metric("年化波动率", f"{backtest['volatility']:.2f}%")
        
        # 资金曲线
        if 'equity_curve' in backtest:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=backtest['equity_curve'],
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
        
        # 分析建议
        if 'analysis' in results:
            analysis = results['analysis']
            if 'recommendations' in analysis:
                st.subheader("💡 投资建议")
                for i, rec in enumerate(analysis['recommendations'], 1):
                    st.write(f"{i}. {rec}")

def show_ml_interface():
    """机器学习界面"""
    st.subheader("🤖 机器学习预测")
    
    tab1, tab2, tab3 = st.tabs(["🎯 模型训练", "📊 预测分析", "⚡ 模型优化"])
    
    with tab1:
        st.markdown("#### 模型配置")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            symbol = st.text_input("股票代码", "AAPL", key="ml_symbol")
            model_type = st.selectbox("模型类型", [
                "xgboost", "random_forest", "lstm", "linear_regression"
            ])
            
        with col2:
            target_days = st.slider("预测天数", 1, 30, 5)
            train_days = st.slider("训练天数", 100, 1000, 365)
        
        if st.button("🎯 训练模型", type="primary"):
            with st.spinner("正在训练模型..."):
                try:
                    system = initialize_system()
                    if system:
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=train_days)).strftime('%Y-%m-%d')
                        
                        ml_results = system.ml_engine.run_ml_pipeline(
                            symbol, start_date, end_date, model_type, target_days
                        )
                        
                        if ml_results['status'] == 'success':
                            st.success("模型训练完成!")
                            show_ml_results(ml_results)
                        else:
                            st.error(f"训练失败: {ml_results.get('message', '未知错误')}")
                            
                except Exception as e:
                    st.error(f"训练失败: {e}")
    
    with tab2:
        st.markdown("#### 预测结果分析")
        st.info("请先在模型训练标签页训练模型")
    
    with tab3:
        st.markdown("#### 模型优化")
        st.info("模型优化功能开发中...")

def show_ml_results(ml_results):
    """显示ML结果"""
    # 模型性能指标
    performance = ml_results['model_performance']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("准确率", f"{performance['accuracy']:.3f}")
    with col2:
        st.metric("精确率", f"{performance['precision']:.3f}")
    with col3:
        st.metric("召回率", f"{performance['recall']:.3f}")
    with col4:
        st.metric("F1分数", f"{performance['f1_score']:.3f}")
    
    # 特征重要性
    if 'feature_importance' in ml_results:
        st.subheader("📊 特征重要性")
        
        importance = ml_results['feature_importance']
        features = list(importance.keys())
        values = list(importance.values())
        
        fig = px.bar(
            x=values,
            y=features,
            orientation='h',
            title="特征重要性排序"
        )
        
        fig.update_layout(
            height=300,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_deployment_guide():
    """部署指南"""
    st.subheader("🚀 部署指南")
    
    # GitHub部署按钮
    st.markdown("""
    <div class="success-banner">
        <h3>🎯 一键部署到Streamlit Cloud</h3>
        <p>点击下方按钮直接部署到云端，完全免费！</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    [![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy)
    """)
    
    # 部署步骤
    st.markdown("#### 📋 部署步骤")
    
    steps = [
        "1. **Fork项目** - 将项目fork到你的GitHub账户",
        "2. **创建仓库** - 在GitHub创建新仓库并上传代码",
        "3. **连接Streamlit** - 访问 [share.streamlit.io](https://share.streamlit.io) 并连接GitHub",
        "4. **选择仓库** - 选择你的项目仓库",
        "5. **配置部署** - 设置主文件为 `streamlit_app.py`",
        "6. **启动部署** - 点击Deploy开始自动部署"
    ]
    
    for step in steps:
        st.markdown(step)
    
    # 本地运行
    st.markdown("#### 💻 本地运行")
    
    st.code("""
# 克隆项目
git clone https://github.com/yourusername/quant-system.git
cd quant-system

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run streamlit_app.py
    """, language="bash")
    
    # 环境要求
    st.markdown("#### ⚙️ 环境要求")
    
    requirements = {
        "Python": "≥ 3.8",
        "Streamlit": "≥ 1.22",
        "Pandas": "≥ 2.0",
        "NumPy": "≥ 1.24",
        "Plotly": "≥ 5.15"
    }
    
    for req, version in requirements.items():
        st.write(f"• **{req}**: {version}")
    
    # 部署状态
    st.markdown("#### 📊 部署状态")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("部署状态", "🟢 在线", help="系统当前运行状态")
    with col2:
        st.metric("响应时间", "< 2s", help="平均页面加载时间")
    with col3:
        st.metric("可用性", "99.9%", help="系统可用性统计")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>QuantAI Trader</strong> | 智能量化交易系统 | 
    <a href="https://github.com" target="_blank">GitHub</a> | 
    <a href="https://streamlit.io" target="_blank">Streamlit</a></p>
    <p style='font-size: 0.8rem;'>⚠️ 投资有风险，入市需谨慎。本系统仅用于学习和研究目的。</p>
</div>
""", unsafe_allow_html=True)

# 侧边栏信息
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 📞 技术支持")
    st.info("遇到问题？访问我们的GitHub仓库获取帮助")
    
    st.markdown("#### 💡 功能亮点")
    st.success("✅ 多策略回测")
    st.success("✅ AI智能预测") 
    st.success("✅ 实时数据")
    st.success("✅ 云端部署")
    
    if st.button("🔄 重置系统"):
        st.session_state.clear()
        st.success("系统已重置")
        st.experimental_rerun() 

# 主程序入口
if __name__ == "__main__":
    main() 