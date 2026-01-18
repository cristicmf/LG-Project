# LG-Project

## 项目简介

LG-Project 是一个基于 LangGraph 概念的智能代理系统项目，包含邮件代理和计算器代理两个主要组件。本项目旨在展示如何使用 LLM（大型语言模型）来构建智能代理系统，处理各种用户请求。

## 功能特性

### 1. 邮件代理系统 (email_agent.py)

- **邮件分类**：使用 LLM 或规则引擎对邮件进行分类，识别邮件意图、紧急程度和主题
- **文档搜索**：根据邮件分类结果搜索相关文档
- **智能回复**：基于邮件内容和搜索结果生成专业的回复
- **人工审核**：高优先级邮件自动路由到人工审核流程
- **双模式运行**：支持虚拟 LLM（规则引擎）和真实 LLM（OpenAI API）两种运行模式

### 2. 计算器代理系统 (calculator_agent.py)

- **数学计算**：支持基本的数学运算
- **表达式解析**：解析和计算数学表达式

## 技术架构

### 系统架构

本项目采用基于状态的代理图架构，灵感来自 LangGraph 文档：

1. **状态管理**：使用类-based 状态管理，兼容 Python 3.8+
2. **节点函数**：每个功能模块作为独立的节点函数
3. **条件路由**：基于状态条件的动态路由
4. **模块化设计**：清晰的职责分离和代码组织

### 邮件代理系统架构

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
python3 email_agent.py
```

此模式使用规则引擎进行邮件分类和回复生成，不需要 OpenAI API 密钥。

#### 2. 真实 LLM 模式

```bash
python3 email_agent.py 1
```

此模式使用 OpenAI API 进行邮件分类和回复生成，需要在 `.env` 文件中配置有效的 API 密钥。

### 运行计算器代理系统

```bash
python3 calculator_agent.py
```

## 代码结构

```
LG-Project/
├── .env              # 环境变量配置文件
├── README.md         # 项目说明文档
├── email_agent.py    # 邮件代理系统
├── calculator_agent.py # 计算器代理系统
├── main.py           # 主程序入口
├── simple_calculator.py # 简单计算器实现
└── test_openai.py    # OpenAI 库测试脚本
```

### 邮件代理系统核心组件

1. **状态管理**：`EmailAgentState` 类，管理代理系统的状态
2. **节点函数**：
   - `read_email`：读取和解析邮件内容
   - `classify_intent`：对邮件进行分类
   - `search_docs`：搜索相关文档
   - `draft_response`：生成回复
   - `check_escalation`：检查是否需要人工审核
   - `human_review`：人工审核
   - `send_reply`：发送回复
3. **代理图**：`SimpleGraph` 类，实现基于状态的代理图

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

1. 定义新的节点函数，接收状态参数并返回新状态
2. 在 `build_email_agent` 函数中添加节点到代理图
3. 配置节点之间的连接

### 定制分类规则

修改 `classify_intent` 函数中的备选逻辑部分，添加或修改分类规则。

### 定制回复模板

修改 `draft_response` 函数中的备选逻辑部分，调整回复模板和内容。

## 注意事项

1. **API 密钥安全**：不要将包含真实 API 密钥的 `.env` 文件提交到版本控制系统
2. **API 调用成本**：使用真实 LLM 模式时，注意监控 API 调用次数和成本
3. **错误处理**：系统已实现基本的错误处理，但在生产环境中可能需要更详细的错误处理
4. **性能优化**：对于大量邮件处理，可能需要考虑批处理和并发优化

## 许可证

本项目采用 MIT 许可证。

