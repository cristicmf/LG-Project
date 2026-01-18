#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件代理状态管理模块
"""

class EmailAgentState:
    """邮件代理状态类"""
    def __init__(self, email_content, sender_email, email_id, 
                 classification=None, search_results=None, 
                 customer_history=None, draft_response=None, 
                 messages=None):
        # 原始邮件数据
        self.email_content = email_content
        self.sender_email = sender_email
        self.email_id = email_id

        # 分类结果
        self.classification = classification

        # 搜索/API 结果
        self.search_results = search_results  # 原始文档块列表
        self.customer_history = customer_history  # 来自 CRM 的原始客户数据

        # 生成的内容
        self.draft_response = draft_response
        self.messages = messages

    def __getitem__(self, key):
        """支持字典式访问"""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """支持字典式赋值"""
        setattr(self, key, value)

    def get(self, key, default=None):
        """支持字典式 get 方法"""
        return getattr(self, key, default)

    def copy(self):
        """创建副本"""
        return EmailAgentState(
            email_content=self.email_content,
            sender_email=self.sender_email,
            email_id=self.email_id,
            classification=self.classification,
            search_results=self.search_results,
            customer_history=self.customer_history,
            draft_response=self.draft_response,
            messages=self.messages
        )
