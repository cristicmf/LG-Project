#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
起草回复节点
"""

import os
import sys

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
        logger.info("OpenAI API 已配置，可以使用真实的 LLM 生成回复")
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
                            logger.info("从 .env 文件读取到 OpenAI API 密钥，可以使用真实的 LLM 生成回复")
                            break
        except Exception as e:
            logger.warning("读取 .env 文件失败: %s", str(e))
        
        if not openai_available:
            logger.info("未设置 OPENAI_API_KEY 环境变量，将使用模拟 LLM 生成回复")
except ImportError:
    logger.info("未安装 openai 库，将使用模拟 LLM 生成回复")

@PerformanceMonitor("draft_response")
def draft_response(state):
    """生成适当的回复"""
    classification = state["classification"]
    search_results = state["search_results"]
    email_content = state["email_content"]
    
    if not classification:
        return state
    
    draft = ""
    intent = classification["intent"]
    urgency = classification["urgency"]
    topic = classification.get("topic", "")
    summary = classification.get("summary", "")
    
    # 尝试使用 OpenAI API 生成回复
    if openai_available:
        try:
            # 构建回复生成提示（使用普通字符串连接避免 f-string 转义问题）
            prompt = "请根据以下信息为客户生成一封专业的支持邮件回复：\n\n"
            prompt += "原始邮件内容：\n"
            prompt += email_content + "\n\n"
            prompt += "邮件分类：\n"
            prompt += "- 意图：" + intent + "\n"
            prompt += "- 紧急程度：" + urgency + "\n"
            prompt += "- 主题：" + topic + "\n"
            prompt += "- 摘要：" + summary + "\n\n"
            prompt += "相关文档搜索结果：\n"
            if search_results:
                prompt += " ".join(search_results)
            else:
                prompt += "无"
            prompt += "\n\n"
            prompt += "要求：\n"
            prompt += "1. 使用专业、友好的语气\n"
            prompt += "2. 直接针对客户的问题提供解决方案\n"
            prompt += "3. 包含所有相关的文档信息\n"
            prompt += "4. 根据邮件意图调整回复内容：\n"
            prompt += "   - question: 提供清晰的答案和指导\n"
            prompt += "   - bug: 表达歉意并提供故障排除步骤\n"
            prompt += "   - billing: 表达理解并提供退款或解决账单问题的步骤\n"
            prompt += "   - feature: 感谢反馈并说明功能请求的处理流程\n"
            prompt += "   - complex: 说明将由高级支持工程师跟进\n"
            prompt += "5. 保持回复简洁明了\n"
            prompt += "6. 使用英文回复"
            
            # 调用 OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的客户支持邮件回复助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # 解析 API 响应
            draft = response.choices[0].message.content
            logger.info("使用 OpenAI API 生成回复")
            
        except Exception as e:
            logger.error("使用 OpenAI API 生成回复失败，将使用备选逻辑: %s", str(e))
            # 执行备选逻辑
            # 开头
            draft = "Dear Customer,\n\n"
            
            # 基于意图的回复
            if intent == "question":
                draft += "Thank you for your question. "
                if search_results:
                    draft += "According to our documentation: "
                    draft += " ".join(search_results)
            elif intent == "bug":
                draft += "We're sorry to hear you're experiencing an issue. "
                if search_results:
                    draft += "Please try the following troubleshooting steps: "
                    draft += " ".join(search_results)
                draft += " If the issue persists, please reply with more details."
            elif intent == "billing":
                draft += "Thank you for bringing this billing issue to our attention. "
                if search_results:
                    draft += " ".join(search_results)
                draft += " Please provide your transaction ID so we can investigate further."
            elif intent == "feature":
                draft += "Thank you for your feature request. "
                if search_results:
                    draft += " ".join(search_results)
                draft += " We appreciate your feedback!"
            elif intent == "complex":
                draft += "Thank you for reaching out about this complex issue. "
                draft += "One of our senior support engineers will contact you shortly to assist with this matter."
            
            # 结尾
            draft += "\n\nBest regards,\nCustomer Support Team"
    else:
        # 使用备选逻辑生成回复
        # 开头
        draft = "Dear Customer,\n\n"
        
        # 基于意图的回复
        if intent == "question":
            draft += "Thank you for your question. "
            if search_results:
                draft += "According to our documentation: "
                draft += " ".join(search_results)
        elif intent == "bug":
            draft += "We're sorry to hear you're experiencing an issue. "
            if search_results:
                draft += "Please try the following troubleshooting steps: "
                draft += " ".join(search_results)
            draft += " If the issue persists, please reply with more details."
        elif intent == "billing":
            draft += "Thank you for bringing this billing issue to our attention. "
            if search_results:
                draft += " ".join(search_results)
            draft += " Please provide your transaction ID so we can investigate further."
        elif intent == "feature":
            draft += "Thank you for your feature request. "
            if search_results:
                draft += " ".join(search_results)
            draft += " We appreciate your feedback!"
        elif intent == "complex":
            draft += "Thank you for reaching out about this complex issue. "
            draft += "One of our senior support engineers will contact you shortly to assist with this matter."
        
        # 结尾
        draft += "\n\nBest regards,\nCustomer Support Team"
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["draft_response"] = draft
    return new_state
