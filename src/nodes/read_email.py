#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取邮件节点
"""

import os
import sys

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from monitoring import PerformanceMonitor
from logging_config import get_logger

logger = get_logger()

@PerformanceMonitor("read_email")
def read_email(state):
    """读取和解析邮件内容"""
    # 在实际应用中，这里会从邮件服务器或 API 读取邮件
    # 这里我们假设邮件内容已经在状态中
    logger.info("Email content read successfully")
    return state
