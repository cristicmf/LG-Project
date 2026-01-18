#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档数据模块
"""

# 示例文档库（结构化格式）
EXAMPLE_DOCUMENTS = [
    # 账户管理
    {"content": "如何重置密码：1. 访问登录页面 2. 点击'忘记密码' 3. 按照邮件指示操作", "tags": ["password", "reset", "账户管理", "密码重置"]},
    {"content": "密码重置常见问题：密码重置链接有效期为24小时", "tags": ["password", "reset", "账户管理", "密码重置"]},
    {"content": "如何修改账户信息：登录后进入'个人中心' -> '账户设置'进行修改", "tags": ["account", "profile", "账户管理"]},
    {"content": "账户安全提示：定期更换密码，不要与他人共享账户信息", "tags": ["account", "security", "账户管理"]},
    {"content": "如何绑定手机/邮箱：登录后进入'个人中心' -> '安全设置'进行绑定", "tags": ["account", "security", "账户管理"]},
    
    # 技术支持
    {"content": "常见技术问题排查：1. 清除浏览器缓存 2. 尝试使用不同浏览器 3. 检查网络连接", "tags": ["technical", "troubleshooting", "技术支持"]},
    {"content": "错误代码参考：504 错误通常表示服务器超时", "tags": ["technical", "error", "技术支持"]},
    {"content": "如何更新软件：进入'设置' -> '关于' -> '检查更新'", "tags": ["technical", "update", "技术支持"]},
    {"content": "系统要求：Windows 10+ 或 macOS 10.15+，至少 4GB 内存", "tags": ["technical", "system", "技术支持"]},
    {"content": "如何查看系统日志：进入'设置' -> '高级' -> '系统日志'", "tags": ["technical", "logs", "技术支持"]},
    
    # 账单与支付
    {"content": "账单问题处理：如果您被重复收费，请联系客服提供交易ID", "tags": ["billing", "charge", "账单与支付"]},
    {"content": "退款政策：符合条件的退款将在3-5个工作日内处理", "tags": ["billing", "refund", "账单与支付"]},
    {"content": "支付方式：我们支持信用卡、PayPal、支付宝和微信支付", "tags": ["billing", "payment", "账单与支付"]},
    {"content": "订阅管理：如何取消自动续费：登录后进入'订阅' -> '管理订阅' -> '取消自动续费'", "tags": ["billing", "subscription", "账单与支付"]},
    {"content": "发票开具：如何申请发票：登录后进入'订单' -> '发票管理' -> '申请发票'", "tags": ["billing", "invoice", "账单与支付"]},
    
    # 退货与售后
    {"content": "退货流程：1. 登录账户 2. 进入订单页面 3. 选择要退货的商品 4. 填写退货原因 5. 等待审核 6. 邮寄商品 7. 等待退款", "tags": ["return", "process", "退货与售后"]},
    {"content": "退货政策：商品在收到后7天内可以申请退货，需保持商品完好无损，配件齐全", "tags": ["return", "policy", "退货与售后"]},
    {"content": "退货物流：退货产生的运费由买家承担，除非商品存在质量问题", "tags": ["return", "shipping", "退货与售后"]},
    {"content": "换货流程：1. 登录账户 2. 进入订单页面 3. 选择要换货的商品 4. 填写换货原因 5. 等待审核 6. 邮寄商品 7. 等待新商品发货", "tags": ["exchange", "process", "退货与售后"]},
    {"content": "保修政策：我们提供1年的产品保修服务，不包括人为损坏", "tags": ["warranty", "policy", "退货与售后"]},
    
    # 功能与使用
    {"content": "功能请求流程：我们会定期评估用户提出的功能请求", "tags": ["feature", "request", "功能与使用"]},
    {"content": "即将推出的功能：我们计划在下个季度添加暗黑模式", "tags": ["feature", "upcoming", "功能与使用"]},
    {"content": "如何使用搜索功能：在顶部导航栏的搜索框中输入关键词进行搜索", "tags": ["feature", "search", "功能与使用"]},
    {"content": "如何导出数据：进入'设置' -> '数据' -> '导出数据'", "tags": ["feature", "export", "功能与使用"]},
    {"content": "如何分享内容：点击内容下方的'分享'按钮，选择分享方式", "tags": ["feature", "share", "功能与使用"]},
    
    # 常见问题
    {"content": "如何联系客服：1. 在线客服：工作日 9:00-18:00 2. 邮件：support@example.com 3. 电话：400-123-4567", "tags": ["contact", "support", "常见问题"]},
    {"content": "服务时间：我们的客服团队工作时间为工作日 9:00-18:00，周末休息", "tags": ["service", "hours", "常见问题"]},
    {"content": "如何跟踪订单：登录后进入'订单' -> '查看详情' -> '物流信息'", "tags": ["order", "tracking", "常见问题"]},
    {"content": "如何取消订单：在订单状态为'待处理'时，可以登录后进入'订单' -> '取消订单'", "tags": ["order", "cancel", "常见问题"]},
    {"content": "如何修改订单：在订单状态为'待处理'时，可以联系客服进行修改", "tags": ["order", "modify", "常见问题"]}
]

# 获取文档库函数
def get_documents():
    """获取示例文档库"""
    return EXAMPLE_DOCUMENTS
