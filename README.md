# 智能体检报告解析系统

一个生产级别的智能体检报告解析和 AI 对话系统，支持文件上传、OCR 识别、PDF 解析和结构化信息抽取。
### 7. 启动后端服务

```bash
python app.py
```

后端将在 http://localhost:8000 启动

API 文档：http://localhost:8000/docs

### 8. 启动前端


```bash
python frontend/gradio_app.py
```

前端将在 http://localhost:7860 启动




---

## 📚 详细指南

- **解决冲突**: [fix-conflicts.bat 直接运行](#) ⭐ 最简单
- **零冲突安装**: [INSTALL-CLEAN.md](./INSTALL-CLEAN.md)
- **Python 3.11 通用**: [INSTALL-PY311.md](./INSTALL-PY311.md)

## 功能特性

- 用户注册/登录（JWT 认证）
- 多会话管理
- 文件上传（PDF/图片/TXT）
- **EasyOCR / PaddleOCR 双引擎支持
- PyMuPDF PDF 解析
- AI 驱动的信息抽取
- 动态 JSON 输出
- Gradio 聊天界面

## 技术栈

**后端**
- FastAPI（异步框架）
- SQLAlchemy ORM
- MySQL
- Pydantic

**AI 服务**
- Ollama 本地模型（默认）
- OpenAI API（兼容 DeepSeek、通义千问等）
- PaddleOCR
- PyMuPDF

**前端**
- Gradio

## 项目结构

```
medical_ai/
├── app.py                 # 主入口文件
├── config.py              # 配置文件
├── database.py            # 数据库连接
├── requirements.txt       # 依赖列表
├── init.sql               # 数据库初始化脚本
├── models/                # 数据模型
│   ├── user.py
│   ├── conversation.py
│   └── chat_message.py
├── schemas/               # Pydantic 模式
│   ├── auth.py
│   ├── chat.py
│   └── report.py
├── api/                   # API 路由
│   ├── auth.py
│   ├── chat.py
│   └── history.py
├── services/              # 业务逻辑层
│   ├── ocr_service.py
│   ├── pdf_service.py
│   ├── text_clean_service.py
│   ├── ai_service.py
│   ├── parser_service.py
│   └── conversation_service.py
├── utils/                 # 工具类
│   ├── jwt.py
│   ├── logger.py
│   └── response.py
├── frontend/              # 前端
│   └── gradio_app.py
└── uploads/               # 上传文件目录
```

## 快速开始

### 1. 环境要求

- Python 3.10.13（推荐，兼容性最好）
- MySQL 8.0+

### 2. 安装系统依赖（重要）

#### Windows（安装 Poppler）
```bash
# 下载 Poppler：https://github.com/oschwartz10612/poppler-windows/releases/
# 解压后添加到 PATH，或使用 pip
pip install poppler
```

#### Linux（Ubuntu/Debian）
```bash
sudo apt install poppler-utils
```

### 3. 安装 Python 依赖

使用清华镜像加速安装：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 配置数据库

编辑 `config.py` 或创建 `.env` 文件配置数据库连接：

```python
DATABASE_URL = "mysql+pymysql://root:password@localhost:3306/medical_ai?charset=utf8mb4"
```

### 5. 初始化数据库

在 MySQL 中执行初始化脚本：

```bash
mysql -u root -p < init.sql
```

### 6. 配置 LLM（本地 Ollama）

#### 使用 Ollama（默认）

1. **安装 Ollama**
   - 下载地址：https://ollama.ai/download

2. **拉取模型**
   ```bash
   ollama pull llama3.1
   ```
   或使用其他模型：qwen2、yi、deepseek-coder-v2 等

3. **启动 Ollama 服务**
   ```bash
   ollama serve
   ```

4. **修改 config.py**（默认已配置）
   ```python
   LLM_PROVIDER = "ollama"
   LLM_API_KEY = "ollama"
   LLM_API_BASE = "http://localhost:11434/v1"
   LLM_MODEL = "llama3.1"
   ```

#### 使用 OpenAI 兼容 API（可选）

在 `config.py` 中配置：

```python
LLM_PROVIDER = "openai"
LLM_API_KEY = "your-api-key"
LLM_API_BASE = "https://api.openai.com/v1"
LLM_MODEL = "gpt-3.5-turbo"
```

### 7. 启动后端服务

```bash
python app.py
```

后端将在 http://localhost:8000 启动

API 文档：http://localhost:8000/docs

### 8. 启动前端

新开一个终端：

```bash
python frontend/gradio_app.py
```

前端将在 http://localhost:7860 启动

## API 接口

### 认证

- `POST /api/register` - 用户注册
- `POST /api/login` - 用户登录

### 聊天

- `POST /api/chat` - 发送消息（支持文件上传）

### 历史记录

- `GET /api/conversations` - 获取会话列表
- `GET /api/history/{conversation_id}` - 获取会话历史
- `DELETE /api/conversation/{conversation_id}` - 删除会话

## 使用说明

1. 注册/登录账号
2. 新建会话
3. 上传体检报告（PDF/图片）或输入文本
4. 查看解析后的结构化 JSON
5. 继续对话或创建新会话

## 注意事项

- 首次运行 PaddleOCR 会自动下载模型文件
- 确保 Ollama 服务已启动并已拉取对应模型
- 确保 Poppler 已安装（PDF OCR 需要）
- 确保 MySQL 服务已启动
- 建议使用 Python 3.10.13 以获得最佳兼容性
- 推荐使用 Python 虚拟环境

## 依赖说明

本项目采用**稳定版本收敛策略**：

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| FastAPI | >=0.104.1 | 异步框架 |
| Gradio | >=4.44.0 | 稳定 Chatbot UI |
| PyMuPDF | >=1.26.0 | PDF 解析（预编译 wheel） |
| openai | >=1.40.0 | 新版 API 兼容 |
| pdf2image | >=1.17.0 | PDF 转图片 |

## 许可证

MIT License
