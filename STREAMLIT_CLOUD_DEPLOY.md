# 🚀 Streamlit Cloud 部署指南

## Tony&Associates QuantAI Trader 部署教程

### 第一步: 准备工作
✅ 代码已推送到 GitHub: https://github.com/tonyxuyk/quant-system

### 第二步: 访问 Streamlit Cloud
1. 打开 [share.streamlit.io](https://share.streamlit.io)
2. 使用 GitHub 账号登录

### 第三步: 部署应用
1. 点击 "New app" 按钮
2. 选择 GitHub 仓库: `tonyxuyk/quant-system`
3. 分支选择: `main`
4. 主文件路径: `streamlit_app.py`
5. 应用 URL: 自定义或使用默认

### 第四步: 配置环境变量（可选）
在 Streamlit Cloud 应用设置中添加：
```toml
[general]
environment = "production"
debug = false

[akshare]
# akshare 不需要API密钥

[tushare]
token = "你的tushare令牌"
```

### 第五步: 部署完成
- 部署时间约 2-5 分钟
- 应用将自动可用于访问
- 支持实时更新（代码推送后自动重新部署）

## 📋 部署清单

### ✅ 已完成
- [x] streamlit_app.py 应用文件
- [x] requirements.txt 依赖配置
- [x] runtime.txt Python版本配置
- [x] .streamlit/config.toml 应用配置
- [x] 错误处理和降级方案
- [x] 演示模式支持
- [x] 代码推送到 GitHub

### 🎯 功能特点
- **多市场支持**: A股、港股、美股
- **策略回测**: 5种量化策略
- **智能选股**: 基于规则的股票筛选
- **数据可视化**: 实时图表和技术指标
- **机器学习**: AI价格预测（开发中）
- **响应式设计**: 适配各种设备

### 🔧 技术架构
- **前端**: Streamlit + Plotly
- **数据源**: Akshare + Tushare
- **后端**: Python 集成系统
- **部署**: Streamlit Cloud
- **版本**: Python 3.11

### 📱 访问方式
部署完成后，应用将可以通过以下方式访问：
- Streamlit Cloud 提供的 URL
- 自定义域名（高级功能）

### 🚨 注意事项
1. **免费版限制**:
   - 1GB 内存限制
   - 共享 CPU 资源
   - 应用休眠机制

2. **数据源依赖**:
   - akshare: 免费数据源，无需配置
   - tushare: 需要注册账号获取token

3. **性能优化**:
   - 使用 @st.cache_resource 缓存系统初始化
   - 使用 @st.cache_data 缓存数据获取
   - 实现降级演示模式

### 📞 技术支持
如有问题，请在 GitHub 仓库提交 Issue:
https://github.com/tonyxuyk/quant-system/issues

---
**部署成功！** 🎉 您的量化交易系统已经可以在云端访问了。 