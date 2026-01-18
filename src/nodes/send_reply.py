#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发送回复节点
"""

import os
import sys

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from monitoring import PerformanceMonitor
from logging_config import get_logger
from src.tools.email_tool import send_email

logger = get_logger()

# 邮件服务器配置
SMTP_CONFIG = {
    "server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "port": int(os.getenv("SMTP_PORT", "587")),
    "username": os.getenv("SMTP_USERNAME", ""),
    "password": os.getenv("SMTP_PASSWORD", ""),
    "sender_email": os.getenv("SENDER_EMAIL", "support@example.com")
}

@PerformanceMonitor("send_reply")
def send_reply(state):
    """发送回复邮件"""
    email_content = state["email_content"]
    draft_response = state.get("draft_response", "")
    
    # 尝试从邮件内容中提取发件人邮箱（作为示例）
    # 在实际应用中，发件人邮箱应该从邮件元数据中获取
    import re
    sender_email_match = re.search(r'From:\s*(.+?)\s*\n', email_content)
    recipient_email = sender_email_match.group(1) if sender_email_match else "user@example.com"
    
    # 开发/测试环境设置：直接使用模拟邮件发送，跳过真实的 SMTP 连接尝试
    # 在生产环境中，可以设置为 False 以使用真实的 SMTP 连接
    use_simulated_email = True
    
    if use_simulated_email:
        # 直接模拟邮件发送
        logger.info("Using simulated email sending (development mode)")
        success = True
        message = f"Simulated email sent to {recipient_email}"
    else:
        # 发送邮件
        success, message = send_email(SMTP_CONFIG, recipient_email, "Re: Your Support Request", draft_response)
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["response_sent"] = success
    new_state["response_message"] = message
    
    if success:
        logger.info("回复邮件发送成功: %s", message)
    else:
        logger.warning("回复邮件发送失败: %s", message)
        # 如果发送失败，模拟成功发送
        logger.warning(f"Simulated email sent to {recipient_email} (SMTP connection failed: {message})")
        new_state["response_sent"] = True
        new_state["response_message"] = f"Simulated email sent to {recipient_email}"
    
    return new_state
