#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类意图节点
"""

import os
import sys
import json

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from monitoring import PerformanceMonitor
from logging_config import get_logger

logger = get_logger()

# 尝试导入 OpenAI API
openai_available = False
try:
    from openai import OpenAI
    # 检查是否设置了 API 密钥
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        openai_available = True
        logger.info("OpenAI API 已配置，可以使用真实的 LLM 进行邮件分类")
    else:
        # 尝试从 .env 文件读取 API 密钥
        try:
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        if key.strip() == 'OPENAI_API_KEY' and value.strip():
                            os.environ["OPENAI_API_KEY"] = value.strip()
                            client = OpenAI(api_key=value.strip())
                            openai_available = True
                            logger.info("从 .env 文件读取到 OpenAI API 密钥，可以使用真实的 LLM 进行邮件分类")
                            break
        except Exception as e:
            logger.warning("读取 .env 文件失败: %s", str(e))
        
        if not openai_available:
            logger.info("未设置 OPENAI_API_KEY 环境变量，将使用模拟 LLM 进行邮件分类")
except ImportError:
    logger.info("未安装 openai 库，将使用模拟 LLM 进行邮件分类")

@PerformanceMonitor("classify_intent")
def classify_intent(state):
    """使用 LLM 对邮件进行分类"""
    email_content = state["email_content"]
    
    # 简单的分类逻辑（作为备选）
    classification = {
        "intent": "question",
        "urgency": "medium",
        "topic": "general",
        "summary": "Customer support request"
    }
    
    # 尝试使用 OpenAI API 进行分类
    if openai_available:
        try:
            # 构建分类提示（使用普通字符串连接避免 f-string 转义问题）
            prompt = "请对以下客户支持邮件进行分类：\n\n"
            prompt += "邮件内容：\n"
            prompt += email_content + "\n\n"
            prompt += "请按照以下格式返回分类结果（使用 JSON 格式）：\n"
            prompt += '{"intent": "question|bug|billing|feature|complex", "urgency": "low|medium|high|critical", "topic": "主题描述", "summary": "邮件摘要"}\n\n'
            prompt += "分类说明：\n"
            prompt += "- intent: 邮件意图\n"
            prompt += "  - question: 一般问题（如密码重置、使用说明等）\n"
            prompt += "  - bug: 技术问题（如崩溃、错误等）\n"
            prompt += "  - billing: 账单问题（如收费、退款等）\n"
            prompt += "  - feature: 功能请求（如添加新功能）\n"
            prompt += "  - complex: 复杂问题（如 API 集成、系统配置等）\n"
            prompt += "- urgency: 紧急程度\n"
            prompt += "  - low: 低优先级\n"
            prompt += "  - medium: 中优先级\n"
            prompt += "  - high: 高优先级\n"
            prompt += "  - critical: 紧急优先级\n"
            prompt += "- topic: 邮件主题（简短描述）\n"
            prompt += "- summary: 邮件摘要（简要总结邮件内容）"
            
            # 调用 OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的客户支持邮件分类助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # 解析 API 响应
            classification = json.loads(response.choices[0].message.content)
            logger.info("使用 OpenAI API 完成邮件分类")
            
        except Exception as e:
            logger.error("使用 OpenAI API 分类失败，将使用备选逻辑: %s", str(e))
            # 执行备选逻辑
            if "password" in email_content.lower():
                classification["intent"] = "question"
                classification["topic"] = "password reset"
            elif "crash" in email_content.lower() or "error" in email_content.lower():
                classification["intent"] = "bug"
                classification["urgency"] = "high"
                classification["topic"] = "technical issue"
            elif "charge" in email_content.lower() or "bill" in email_content.lower():
                classification["intent"] = "billing"
                classification["urgency"] = "high"
                classification["topic"] = "billing issue"
            elif "feature" in email_content.lower() or "add" in email_content.lower():
                classification["intent"] = "feature"
                classification["urgency"] = "low"
                classification["topic"] = "feature request"
            elif "api" in email_content.lower() or "integration" in email_content.lower():
                classification["intent"] = "complex"
                classification["urgency"] = "medium"
                classification["topic"] = "technical integration"
            elif "return" in email_content.lower() or "退货" in email_content:
                classification["intent"] = "question"
                classification["urgency"] = "medium"
                classification["topic"] = "return policy"
    else:
        # 使用备选逻辑进行分类
        if "password" in email_content.lower():
            classification["intent"] = "question"
            classification["topic"] = "password reset"
        elif "crash" in email_content.lower() or "error" in email_content.lower():
            classification["intent"] = "bug"
            classification["urgency"] = "high"
            classification["topic"] = "technical issue"
        elif "charge" in email_content.lower() or "bill" in email_content.lower():
            classification["intent"] = "billing"
            classification["urgency"] = "high"
            classification["topic"] = "billing issue"
        elif "feature" in email_content.lower() or "add" in email_content.lower():
            classification["intent"] = "feature"
            classification["urgency"] = "low"
            classification["topic"] = "feature request"
        elif "api" in email_content.lower() or "integration" in email_content.lower():
            classification["intent"] = "complex"
            classification["urgency"] = "medium"
            classification["topic"] = "technical integration"
        elif "return" in email_content.lower() or "退货" in email_content:
            classification["intent"] = "question"
            classification["urgency"] = "medium"
            classification["topic"] = "return policy"
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["classification"] = classification
    return new_state
