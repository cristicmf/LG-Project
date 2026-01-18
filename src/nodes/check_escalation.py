#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查是否需要人工审核节点
"""

import os
import sys

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from monitoring import PerformanceMonitor
from logging_config import get_logger

logger = get_logger()

@PerformanceMonitor("check_escalation")
def check_escalation(state):
    """决定是否需要人工审核"""
    classification = state["classification"]
    if not classification:
        return "human_review"
    
    # 基于紧急程度和意图决定是否需要人工审核
    urgency = classification["urgency"]
    intent = classification["intent"]
    
    if urgency in ["high", "critical"] or intent == "complex":
        logger.info("邮件需要人工审核: 紧急程度=%s, 意图=%s", urgency, intent)
        return "human_review"
    else:
        logger.info("邮件不需要人工审核: 紧急程度=%s, 意图=%s", urgency, intent)
        return "send_reply"
