#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tony&Associates QuantAI Trader - 主应用入口
量化交易系统主应用
"""

import subprocess
import sys
import os

def main():
    """主函数"""
    print("🚀 启动 Tony&Associates QuantAI Trader")
    print("=" * 50)
    
    try:
        # 确保在正确的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_dir)
        
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        
    except KeyboardInterrupt:
        print("\n✋ 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 