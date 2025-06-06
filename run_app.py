#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tony&Associates QuantAI Trader - 应用启动脚本
智能启动脚本，自动检查依赖并启动Streamlit应用
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ 是必需的")
        logger.error(f"当前版本: {version.major}.{version.minor}.{version.micro}")
        return False
    
    logger.info(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
    return True

def check_and_install_requirements():
    """检查并安装依赖包"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error(f"❌ 找不到requirements.txt文件: {requirements_file}")
        return False
    
    logger.info("🔍 检查依赖包...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "--quiet", "-r", str(requirements_file)
        ])
        logger.info("✅ 所有依赖包安装完成")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 安装依赖包失败: {e}")
        return False
    
    return True

def check_streamlit_config():
    """检查Streamlit配置"""
    config_dir = Path(__file__).parent / ".streamlit"
    config_file = config_dir / "config.toml"
    
    if not config_dir.exists():
        logger.info("📁 创建Streamlit配置目录...")
        config_dir.mkdir(exist_ok=True)
    
    if not config_file.exists():
        logger.info("⚙️ 创建默认Streamlit配置...")
        config_content = """[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#31333f"
font = "sans serif"
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        logger.info("✅ Streamlit配置文件创建完成")
    else:
        logger.info("✅ Streamlit配置文件已存在")
    
    return True

def check_app_files():
    """检查应用文件"""
    app_file = Path(__file__).parent / "streamlit_app.py"
    integration_file = Path(__file__).parent / "integration.py"
    
    missing_files = []
    
    if not app_file.exists():
        missing_files.append("streamlit_app.py")
    
    if not integration_file.exists():
        missing_files.append("integration.py")
    
    if missing_files:
        logger.error(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    logger.info("✅ 应用文件检查通过")
    return True

def start_streamlit():
    """启动Streamlit应用"""
    app_file = Path(__file__).parent / "streamlit_app.py"
    
    try:
        logger.info("🚀 启动Tony&Associates QuantAI Trader...")
        logger.info("📊 应用将在浏览器中自动打开...")
        logger.info("🔗 如果没有自动打开，请访问: http://localhost:8501")
        
        # 启动Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 启动Streamlit失败: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("👋 应用已停止")
        return True
    
    return True

def main():
    """主函数"""
    print("🚀 Tony&Associates QuantAI Trader 启动检查...")
    print("=" * 50)
    
    # 检查步骤
    checks = [
        ("Python版本检查", check_python_version),
        ("依赖包检查", check_and_install_requirements),
        ("Streamlit配置检查", check_streamlit_config),
        ("应用文件检查", check_app_files),
    ]
    
    for name, check_func in checks:
        logger.info(f"🔍 {name}...")
        if not check_func():
            logger.error(f"❌ {name}失败，无法启动应用")
            return False
        logger.info(f"✅ {name}完成")
        print("-" * 30)
    
    print("🎉 所有检查通过！")
    print("=" * 50)
    
    # 启动应用
    return start_streamlit()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.error(f"💥 启动失败: {e}")
        sys.exit(1) 