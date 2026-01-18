#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件代理系统主入口
"""

import os
import sys
import json

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from logging_config import get_logger
from src.graph.simple_graph import SimpleGraph
from src.nodes.read_email import read_email
from src.nodes.classify_intent import classify_intent
from src.nodes.search_docs import search_docs
from src.nodes.draft_response import draft_response
from src.nodes.check_escalation import check_escalation
from src.nodes.human_review import human_review
from src.nodes.send_reply import send_reply

logger = get_logger()

class EmailAgent:
    """邮件代理系统"""
    def __init__(self):
        """初始化邮件代理系统"""
        self.graph = None
        self.initialize_graph()
    
    def initialize_graph(self):
        """初始化工作流图"""
        # 创建简单图实例
        self.graph = SimpleGraph()
        
        # 添加节点
        self.graph.add_node("read_email", read_email)
        self.graph.add_node("classify_intent", classify_intent)
        self.graph.add_node("search_docs", search_docs)
        self.graph.add_node("draft_response", draft_response)
        self.graph.add_node("human_review", human_review)
        self.graph.add_node("send_reply", send_reply)
        
        # 添加边
        self.graph.add_edge("read_email", "classify_intent")
        self.graph.add_edge("classify_intent", "search_docs")
        self.graph.add_edge("search_docs", "draft_response")
        
        # 添加条件边
        def escalation_condition(state):
            from src.nodes.check_escalation import check_escalation
            return check_escalation(state)
        
        self.graph.add_conditional_edge(
            "draft_response",
            escalation_condition,
            {
                "human_review": "human_review",
                "send_reply": "send_reply"
            }
        )
        
        # 添加最终边
        self.graph.add_edge("human_review", "send_reply")
        
        # 设置入口点
        self.graph.set_entry_point("read_email")
        
        # 编译图
        self.graph.compile()
        
        logger.info("Email agent graph initialized successfully")
    
    def process_email(self, email_content):
        """处理邮件"""
        # 创建初始状态
        initial_state = {
            "email_content": email_content
        }
        
        # 执行图
        result = self.graph.invoke(initial_state)
        
        logger.info("Email processed successfully")
        return result

if __name__ == "__main__":
    """主函数"""
    # 创建邮件代理实例
    agent = EmailAgent()
    
    # 示例邮件内容
    sample_email = """Subject: Password Reset Request

Hello,

I'm having trouble resetting my password. Can you help me with this?

Thanks,
John Doe"""
    
    # 处理示例邮件
    result = agent.process_email(sample_email)
    
    # 打印结果
    print("Processing result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
