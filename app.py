#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tony&Associates QuantAI Trader - Main Entry Point
主应用入口文件
"""

import streamlit as st
import sys
import os

# 添加项目路径到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入并运行主应用
if __name__ == "__main__":
    try:
        from streamlit_app import main
        main()
    except ImportError as e:
        st.error(f"导入错误: {e}")
        st.error("请确保所有依赖已正确安装: pip install -r requirements.txt")
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}") 