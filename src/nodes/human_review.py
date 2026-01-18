#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工审核节点
"""

import os
import sys

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from monitoring import PerformanceMonitor
from logging_config import get_logger

logger = get_logger()

@PerformanceMonitor("human_review")
def human_review(state):
    """标记邮件为需要人工审核"""
    logger.info("邮件已标记为需要人工审核")
    # 在实际应用中，这里会将邮件发送给人工审核队列
    # 这里我们只是简单地标记状态
    new_state = state.copy()
    new_state["needs_review"] = True
    return new_state
