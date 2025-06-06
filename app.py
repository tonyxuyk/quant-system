#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tony&Associates QuantAI Trader - 应用入口
量化交易系统 - Streamlit Cloud部署版本
"""

# 直接导入主应用
try:
    # 运行主应用
    import streamlit_app
    if hasattr(streamlit_app, 'main'):
        streamlit_app.main()
    else:
        import streamlit as st
        st.error("应用初始化失败，请检查配置")
except Exception as e:
    import streamlit as st
    st.error(f"应用启动失败: {e}")
    st.info("正在尝试修复...")
    
    # 简化版应用
    st.title("📈 Tony&Associates QuantAI Trader")
    st.markdown("**量化交易系统正在启动中...**")
    
    if st.button("🔄 重新加载"):
        st.rerun() 