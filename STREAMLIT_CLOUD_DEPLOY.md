# Streamlit Cloud 部署指南

## 🚀 快速部署到 Streamlit Cloud

### 1. 准备工作
确保您的GitHub仓库包含以下文件：
- ✅ `streamlit_app.py` - 主应用文件
- ✅ `requirements.txt` - 依赖列表
- ✅ `.streamlit/config.toml` - 配置文件

### 2. 部署步骤

#### 步骤1: 访问 Streamlit Cloud
1. 打开 [share.streamlit.io](https://share.streamlit.io)
2. 使用GitHub账号登录

#### 步骤2: 创建新应用
1. 点击 "New app"
2. 选择 "From existing repo"
3. 填写仓库信息：
   - **Repository**: `tonyxuyk/quant-system`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`

#### 步骤3: 高级设置（可选）
```
Python version: 3.11
```

#### 步骤4: 部署
1. 点击 "Deploy!"
2. 等待部署完成（通常需要2-5分钟）

### 3. 依赖说明

当前 `requirements.txt` 包含：
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
requests>=2.31.0
python-dateutil>=2.8.0
pytz>=2023.3
setuptools>=65.0.0
PyYAML>=6.0
akshare>=1.12.0
tushare>=1.2.89
matplotlib>=3.7.0
openpyxl>=3.1.0
lxml>=4.9.0
beautifulsoup4>=4.12.0
```

### 4. 可能的问题和解决方案

#### 问题1: 依赖安装失败
**解决方案**: 
- 检查 `requirements.txt` 格式
- 移除有问题的依赖包
- 使用更宽松的版本要求

#### 问题2: 内存不足
**解决方案**:
- 优化数据加载
- 使用 `@st.cache_data` 缓存
- 减少同时加载的数据量

#### 问题3: 网络请求超时
**解决方案**:
- 增加请求超时时间
- 添加重试机制
- 使用异步请求

### 5. 本地测试

部署前建议本地测试：
```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run streamlit_app.py --server.port 8501
```

### 6. 环境变量配置

如果需要API密钥等敏感信息：
1. 在Streamlit Cloud中设置 Secrets
2. 在应用中使用 `st.secrets`

示例：
```python
# 在 .streamlit/secrets.toml 中
[api]
tushare_token = "your_token_here"

# 在代码中使用
token = st.secrets["api"]["tushare_token"]
```

### 7. 自定义域名（可选）

Streamlit Cloud提供免费子域名：
- 格式: `https://your-app-name.streamlit.app`
- 可以配置自定义域名（需要付费计划）

### 8. 监控和日志

- 在Streamlit Cloud控制台查看部署日志
- 使用 `st.write()` 进行调试输出
- 监控应用性能和错误

### 9. 更新部署

代码更新后：
1. 推送到GitHub
2. Streamlit Cloud会自动重新部署
3. 也可以手动触发重新部署

### 10. 故障排除

#### 常见错误：
1. **ModuleNotFoundError**: 检查 requirements.txt
2. **Memory Error**: 优化数据处理
3. **Timeout Error**: 检查网络请求

#### 调试技巧：
- 查看部署日志
- 本地复现问题
- 逐步注释代码定位问题

---

## 🎯 部署检查清单

- [ ] GitHub仓库已更新
- [ ] requirements.txt 包含所有依赖
- [ ] 本地测试通过
- [ ] 配置文件正确
- [ ] 敏感信息已配置为 Secrets
- [ ] 应用启动正常

## 📞 支持

如遇到问题：
1. 查看 [Streamlit 文档](https://docs.streamlit.io)
2. 检查 [Streamlit Community](https://discuss.streamlit.io)
3. 查看应用日志进行调试

---

**最后更新**: 2025-06-06
**状态**: ✅ 准备就绪 