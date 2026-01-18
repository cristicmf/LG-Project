#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 邮件代理测试用例

from email_agent import EmailAgentState

# 测试用例 1: 密码重置问题
test_email_1 = EmailAgentState(
    email_content="Hello, I forgot my password and need to reset it. Can you help?",
    sender_email="user1@example.com",
    email_id="123"
)

# 测试用例 2: 技术问题
test_email_2 = EmailAgentState(
    email_content="The export feature crashes when I select PDF format. Please help!",
    sender_email="user2@example.com",
    email_id="456"
)

# 测试用例 3: 账单问题
test_email_3 = EmailAgentState(
    email_content="I was charged twice for my subscription. Please refund the extra amount.",
    sender_email="user3@example.com",
    email_id="789"
)

# 测试用例 4: 退货问题
test_email_4= EmailAgentState(
    email_content="你好，我想进行退货，请问需要走什么流程",
    sender_email="user4@example.com",
    email_id="789"
)