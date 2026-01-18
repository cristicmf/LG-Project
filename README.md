# LG-Project

## 项目简介

LG-Project 是一个基于 LangGraph 概念的智能代理系统项目，主要包含邮件代理系统。本项目旨在展示如何使用 LLM（大型语言模型）来构建智能代理系统，处理各种用户请求。

## 功能特性

### 邮件代理系统 (email_agent.py)

- **邮件分类**：使用 LLM 或规则引擎对邮件进行分类，识别邮件意图、紧急程度和主题
- **文档搜索**：根据邮件分类结果搜索相关文档
- **智能回复**：基于邮件内容和搜索结果生成专业的回复
- **人工审核**：高优先级邮件自动路由到人工审核流程
- **双模式运行**：支持虚拟 LLM（规则引擎）和真实 LLM（OpenAI API）两种运行模式

## 技术架构

### 系统架构

本项目采用基于状态的代理图架构，灵感来自 LangGraph 文档，并进行了分层架构设计：

1. **分层架构**：将系统分为核心层、工具层、数据层、图层、节点层和应用层
2. **状态管理**：使用类-based 状态管理，兼容 Python 3.8+
3. **节点函数**：每个功能模块作为独立的节点函数
4. **条件路由**：基于状态条件的动态路由
5. **模块化设计**：清晰的职责分离和代码组织

### 邮件代理系统分层架构

```
LG-Project/
├── src/
│   ├── core/         # 核心层：状态管理
│   ├── tools/        # 工具层：搜索工具、邮件工具
│   ├── data/         # 数据层：示例文档库
│   ├── graph/        # 图层：简单图实现
│   ├── nodes/        # 节点层：功能节点
│   └── app/          # 应用层：邮件代理主入口
├── app.py            # Flask Web 应用
├── evaluate_agent.py # 邮件代理评估器
├── logging_config.py # 日志配置
├── monitoring.py     # 性能监控
└── README.md         # 项目说明文档
```

### 节点流程图

```
┌─────────────┐     ┌───────────────┐     ┌─────────────┐     ┌────────────────┐
│ 读取邮件节点 │ ──> │ 分类意图节点 │ ──> │ 文档搜索节点 │ ──> │ 起草回复节点 │
└─────────────┘     └───────────────┘     └─────────────┘     └────────────────┘
                                                                     │
                                                                     ▼
                                                      ┌───────────────────────────┐
                                                      │ 检查是否需要人工审核节点 │
                                                      └───────────────────────────┘
                                                                     │
                                 ┌─────────────────────┬─────────────┼─────────────────────┐
                                 ▼                     ▼             ▼                     ▼
                      ┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
                      │ 人工审核节点 │     │ 发送回复节点 │     │ 复杂处理节点 │
                      └─────────────────┘     └──────────────┘     └─────────────────┘
                                 └─────────────┬─────────────┘
                                               ▼
                                        ┌──────────────┐
                                        │ 发送回复节点 │
                                        └──────────────┘
```

## 安装和配置

### 环境要求

- Python 3.8+
- pip 包管理工具

### 安装依赖

```bash
# 安装 OpenAI 库（用于真实 LLM 模式）
pip install openai
```

### 配置 OpenAI API 密钥

1. 在项目根目录创建 `.env` 文件
2. 在 `.env` 文件中添加以下内容：

```env
# OpenAI API 密钥
OPENAI_API_KEY=sk-...
```

> 注意：将 `sk-...` 替换为您的实际 OpenAI API 密钥

## 使用方法

### 运行邮件代理系统

#### 1. 虚拟 LLM 模式（默认）

```bash
python3 src/app/email_agent.py
```

此模式使用规则引擎进行邮件分类和回复生成，不需要 OpenAI API 密钥。

#### 2. 真实 LLM 模式

在 `.env` 文件中配置有效的 OpenAI API 密钥后，运行以下命令：

```bash
python3 src/app/email_agent.py
```

系统会自动检测 API 密钥并使用真实 LLM 进行邮件分类和回复生成。

### 运行 Web 界面

```bash
python3 app.py
```

然后在浏览器中访问 `http://localhost:5000` 即可使用 Web 界面。



## 代码结构

```
LG-Project/
├── src/
│   ├── core/         # 核心层：状态管理
│   │   └── state.py  # 邮件代理状态管理
│   ├── tools/        # 工具层：搜索工具、邮件工具
│   │   ├── search_tool.py # 搜索工具
│   │   └── email_tool.py  # 邮件工具
│   ├── data/         # 数据层：示例文档库
│   │   └── documents.py   # 示例文档库
│   ├── graph/        # 图层：简单图实现
│   │   └── simple_graph.py # 简单图实现
│   ├── nodes/        # 节点层：功能节点
│   │   ├── read_email.py        # 读取邮件节点
│   │   ├── classify_intent.py   # 分类意图节点
│   │   ├── search_docs.py       # 文档搜索节点
│   │   ├── draft_response.py    # 起草回复节点
│   │   ├── check_escalation.py  # 检查是否需要人工审核节点
│   │   ├── human_review.py      # 人工审核节点
│   │   └── send_reply.py        # 发送回复节点
│   └── app/          # 应用层：邮件代理主入口
│       └── email_agent.py # 邮件代理主入口
├── app.py            # Flask Web 应用
├── evaluate_agent.py # 邮件代理评估器
├── logging_config.py # 日志配置
├── monitoring.py     # 性能监控
├── requirements.txt  # 依赖项配置
└── README.md         # 项目说明文档
```

### 邮件代理系统核心组件

1. **核心层**：
   - `EmailAgentState` 类：管理代理系统的状态，提供字典式访问和状态复制功能

2. **工具层**：
   - `search_tool.py`：提供基于相似度的搜索功能，用于搜索相关文档
   - `email_tool.py`：提供邮件发送和接收功能

3. **数据层**：
   - `documents.py`：提供示例文档库，用于邮件代理系统的文档搜索

4. **图层**：
   - `simple_graph.py`：实现简单的图结构，模拟 LangGraph 的核心功能，支持节点添加、边添加和条件路由

5. **节点层**：
   - `read_email.py`：读取和解析邮件内容
   - `classify_intent.py`：对邮件进行分类，识别邮件意图、紧急程度和主题
   - `search_docs.py`：根据邮件分类结果搜索相关文档
   - `draft_response.py`：基于邮件内容和搜索结果生成专业的回复
   - `check_escalation.py`：检查是否需要人工审核
   - `human_review.py`：标记邮件为需要人工审核
   - `send_reply.py`：发送回复邮件

6. **应用层**：
   - `email_agent.py`：邮件代理系统主入口，初始化和运行整个系统

7. **其他组件**：
   - `app.py`：Flask Web 应用，提供系统状态、邮件提交和代理评估的 API 接口
   - `evaluate_agent.py`：邮件代理评估器，用于评估邮件代理的性能和准确性
   - `logging_config.py`：日志配置，提供结构化的 JSON 日志
   - `monitoring.py`：性能监控，提供函数执行时间和成功率的监控
   - `requirements.txt`：依赖项配置，列出项目所需的 Python 包

## 运行模式说明

### 虚拟 LLM 模式

- **优点**：不需要 API 密钥，运行速度快
- **缺点**：分类和回复质量有限
- **适用场景**：开发和测试环境

### 真实 LLM 模式

- **优点**：分类和回复质量高，更智能
- **缺点**：需要 API 密钥，有调用成本
- **适用场景**：生产环境，需要高质量回复的场景

## 示例

### 邮件代理系统示例

#### 输入：密码重置请求

```
Hello, I forgot my password and need to reset it. Can you help?
```

#### 输出：智能回复

```
Subject: Password Reset Assistance

Dear Customer,

Thank you for reaching out. I understand that you have forgotten your password and need to reset it. I'm here to help you with that.

To reset your password, please follow these steps:
1. Go to the login page of our website.
2. Click on the "Forgot Password" link.
3. Enter your email address associated with your account.
4. Follow the instructions in the password reset email that will be sent to you.

If you encounter any difficulties during this process, please don't hesitate to contact us for further assistance.

Should you require any additional support, feel free to reach out to us.

Best regards,

Customer Support Team
```

## 扩展和定制

### 添加新节点

1. 在 `src/nodes/` 目录下创建新的节点文件，例如 `new_node.py`
2. 定义新的节点函数，接收状态参数并返回新状态
3. 在 `src/app/email_agent.py` 文件中导入新节点并将其添加到代理图
4. 配置新节点与其他节点之间的连接

### 定制分类规则

修改 `src/nodes/classify_intent.py` 文件中的备选逻辑部分，添加或修改分类规则。

### 定制回复模板

修改 `src/nodes/draft_response.py` 文件中的备选逻辑部分，调整回复模板和内容。

### 添加新工具

1. 在 `src/tools/` 目录下创建新的工具文件，例如 `new_tool.py`
2. 实现新工具的功能
3. 在需要使用该工具的节点文件中导入并使用

### 扩展文档库

修改 `src/data/documents.py` 文件，添加新的示例文档到 `EXAMPLE_DOCUMENTS` 列表中。

## 注意事项

1. **API 密钥安全**：不要将包含真实 API 密钥的 `.env` 文件提交到版本控制系统
2. **API 调用成本**：使用真实 LLM 模式时，注意监控 API 调用次数和成本
3. **错误处理**：系统已实现基本的错误处理，但在生产环境中可能需要更详细的错误处理
4. **性能优化**：对于大量邮件处理，可能需要考虑批处理和并发优化

## 许可证

本项目采用 MIT 许可证。

