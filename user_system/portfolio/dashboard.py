import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

class UserDashboard:
    def __init__(
        self,
        auth_service,
        position_service,
        pick_service,
        strategy_service,
        performance_service
    ):
        self.auth = auth_service
        self.position = position_service
        self.pick = pick_service
        self.strategy = strategy_service
        self.performance = performance_service
    
    async def render_dashboard(self, user_id: UUID):
        st.title("投资工作台")
        
        # 创建选项卡
        tab_names = ["投资组合", "选股记录", "回测历史", "绩效分析"]
        portfolio, picks, backtests, performance = st.tabs(tab_names)
        
        # 投资组合标签页
        with portfolio:
            await self._render_portfolio_tab(user_id)
        
        # 选股记录标签页
        with picks:
            await self._render_picks_tab(user_id)
        
        # 回测历史标签页
        with backtests:
            await self._render_backtests_tab(user_id)
        
        # 绩效分析标签页
        with performance:
            await self._render_performance_tab(user_id)
    
    async def _render_portfolio_tab(self, user_id: UUID):
        st.subheader("当前持仓")
        
        # 添加新持仓的表单
        with st.form("new_position"):
            st.write("添加新持仓")
            cols = st.columns(4)
            with cols[0]:
                symbol = st.text_input("股票代码")
            with cols[1]:
                quantity = st.number_input("数量", min_value=0.0)
            with cols[2]:
                entry_price = st.number_input("买入价格", min_value=0.0)
            with cols[3]:
                entry_date = st.date_input("买入日期")
            
            if st.form_submit_button("添加"):
                try:
                    await self.position.open_position(
                        user_id=user_id,
                        symbol=symbol,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        quantity=quantity
                    )
                    st.success("持仓添加成功！")
                except Exception as e:
                    st.error(f"添加失败：{str(e)}")
        
        # 显示当前持仓
        positions = await self.position.get_current_positions(user_id)
        if positions:
            # 创建持仓数据表格
            position_data = []
            for pos in positions:
                position_data.append({
                    "股票代码": pos.symbol,
                    "买入日期": pos.entry_date,
                    "买入价": f"¥{pos.entry_price:.2f}",
                    "数量": pos.quantity,
                    "现价": f"¥{pos.current_price:.2f}",
                    "市值": f"¥{pos.market_value:.2f}",
                    "收益率": f"{pos.unrealized_gain:.2f}%"
                })
            
            st.dataframe(
                position_data,
                column_config={
                    "收益率": st.column_config.NumberColumn(
                        format="%.2f%%",
                        help="未实现收益率"
                    )
                },
                hide_index=True
            )
            
            # 绘制持仓分布饼图
            fig = px.pie(
                position_data,
                values="市值",
                names="股票代码",
                title="持仓分布"
            )
            st.plotly_chart(fig)
        else:
            st.info("暂无持仓")
    
    async def _render_picks_tab(self, user_id: UUID):
        st.subheader("选股记录")
        
        # 添加新选股的表单
        with st.form("new_pick"):
            st.write("添加选股记录")
            cols = st.columns(3)
            with cols[0]:
                symbol = st.text_input("股票代码")
            with cols[1]:
                reason = st.selectbox(
                    "选股理由",
                    ["技术面", "基本面", "消息面", "其他"]
                )
            with cols[2]:
                expected_gain = st.number_input(
                    "预期收益率(%)",
                    min_value=-100.0,
                    max_value=1000.0
                )
            
            notes = st.text_area("备注")
            
            if st.form_submit_button("添加"):
                try:
                    await self.pick.create_pick(
                        user_id=user_id,
                        symbol=symbol,
                        pick_date=datetime.now().date(),
                        reason=reason,
                        expected_gain=expected_gain,
                        notes=notes
                    )
                    st.success("选股记录添加成功！")
                except Exception as e:
                    st.error(f"添加失败：{str(e)}")
        
        # 显示选股记录
        status_filter = st.selectbox(
            "状态过滤",
            ["全部", "watching", "holding", "sold", "expired"]
        )
        
        picks = await self.pick.get_user_picks(
            user_id,
            status=None if status_filter == "全部" else status_filter
        )
        
        if picks:
            pick_data = []
            for pick in picks:
                pick_data.append({
                    "股票代码": pick.symbol,
                    "选股日期": pick.pick_date,
                    "状态": pick.status,
                    "预期收益": f"{pick.expected_gain:.2f}%" if pick.expected_gain else "-",
                    "实际收益": f"{pick.actual_gain:.2f}%" if pick.actual_gain else "-",
                    "备注": pick.notes or "-"
                })
            
            st.dataframe(
                pick_data,
                hide_index=True
            )
        else:
            st.info("暂无选股记录")
    
    async def _render_backtests_tab(self, user_id: UUID):
        st.subheader("回测历史")
        
        # 获取保存的回测记录
        backtests = await self.strategy.get_user_backtests(user_id)
        
        if backtests:
            backtest_data = []
            for bt in backtests:
                backtest_data.append({
                    "策略名称": bt.strategy_name,
                    "开始日期": bt.start_date,
                    "结束日期": bt.end_date,
                    "初始资金": f"¥{bt.initial_capital:,.2f}",
                    "最终价值": f"¥{bt.final_value:,.2f}",
                    "夏普比率": f"{bt.sharpe_ratio:.2f}",
                    "保存时间": bt.created_at
                })
            
            st.dataframe(
                backtest_data,
                hide_index=True
            )
            
            # 绘制回测业绩对比图
            fig = go.Figure()
            for bt in backtests:
                returns = (bt.final_value / bt.initial_capital - 1) * 100
                fig.add_trace(go.Bar(
                    x=[bt.strategy_name],
                    y=[returns],
                    name=bt.strategy_name
                ))
            
            fig.update_layout(
                title="策略回测业绩对比",
                yaxis_title="收益率(%)",
                showlegend=True
            )
            st.plotly_chart(fig)
        else:
            st.info("暂无保存的回测记录")
    
    async def _render_performance_tab(self, user_id: UUID):
        st.subheader("绩效分析")
        
        # 时间范围选择
        cols = st.columns(2)
        with cols[0]:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now().date() - timedelta(days=365)
            )
        with cols[1]:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now().date()
            )
        
        # 获取绩效指标
        portfolio_stats = await self.performance.calculate_portfolio_performance(
            user_id, start_date, end_date
        )
        
        # 显示关键指标
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("总交易次数", portfolio_stats["total_trades"])
        with metric_cols[1]:
            st.metric("胜率", f"{portfolio_stats['win_rate']:.2f}%")
        with metric_cols[2]:
            st.metric("平均收益", f"{portfolio_stats['avg_return']:.2f}%")
        with metric_cols[3]:
            st.metric("夏普比率", f"{portfolio_stats['sharpe_ratio']:.2f}")
        
        # 获取投资组合历史数据
        history = await self.performance.get_portfolio_history(user_id)
        if history:
            # 绘制资产曲线
            df_history = pd.DataFrame(history)
            fig = px.line(
                df_history,
                x="date",
                y="value",
                title="投资组合价值变化"
            )
            st.plotly_chart(fig)
            
            # 行业配置分析
            sector_allocation = await self.performance.get_sector_allocation(user_id)
            if sector_allocation:
                fig = px.pie(
                    sector_allocation,
                    values="percentage",
                    names="sector",
                    title="行业配置"
                )
                st.plotly_chart(fig)
        
        # 风险指标
        risk_metrics = await self.performance.get_risk_metrics(user_id)
        st.subheader("风险指标")
        
        risk_cols = st.columns(3)
        with risk_cols[0]:
            st.metric("波动率", f"{risk_metrics['volatility']:.2f}%")
        with risk_cols[1]:
            st.metric("VaR(95%)", f"{risk_metrics['var_95']:.2f}%")
        with risk_cols[2]:
            st.metric(
                "最大回撤",
                f"{portfolio_stats['max_drawdown']:.2f}%"
            )
