#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件代理 Web 界面
使用 Flask 框架提供用户友好的界面
"""

import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# 导入配置
from config import SMTP_CONFIG, IMAP_CONFIG, APP_CONFIG

# 导入监控模块
from monitoring import get_performance_metrics, get_error_counts, get_function_calls, log_system_status

# 导入评估模块
from evaluate_agent import AgentEvaluator, get_sample_test_cases

# 导入邮件代理
from email_agent import build_email_agent

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 创建邮件代理
email_agent = build_email_agent()

# 根路由
@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

# 系统状态路由
@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    """获取系统状态"""
    # 记录系统状态
    system_status = log_system_status()
    
    # 获取性能指标
    performance_metrics = get_performance_metrics()
    
    # 获取错误计数
    error_counts = get_error_counts()
    
    # 获取函数调用次数
    function_calls = get_function_calls()
    
    return jsonify({
        "system_status": system_status,
        "performance_metrics": performance_metrics,
        "error_counts": error_counts,
        "function_calls": function_calls
    })

# 提交邮件请求路由
@app.route('/api/submit-email', methods=['POST'])
def submit_email():
    """提交邮件请求"""
    try:
        # 获取请求数据
        data = request.json
        email_content = data.get('email_content')
        sender_email = data.get('sender_email')
        
        if not email_content or not sender_email:
            return jsonify({"error": "缺少必要的邮件内容或发件人邮箱"}), 400
        
        # 创建邮件请求
        email_request = {
            "email_id": str(int(datetime.now().timestamp())),
            "sender_email": sender_email,
            "email_content": email_content
        }
        
        # 执行邮件代理
        result = email_agent.invoke(email_request)
        
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 评估邮件代理路由
@app.route('/api/evaluate-agent', methods=['POST'])
def evaluate_agent():
    """评估邮件代理"""
    try:
        # 创建评估器
        evaluator = AgentEvaluator()
        
        # 获取示例测试用例
        test_cases = get_sample_test_cases()
        
        # 评估邮件代理
        metrics = evaluator.evaluate(test_cases, evaluate_accuracy=True)
        
        # 获取评估结果
        results = evaluator.get_evaluation_results()
        
        return jsonify({"success": True, "metrics": metrics, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 静态文件路由
@app.route('/static/<path:path>')
def static_file(path):
    """提供静态文件"""
    return app.send_static_file(path)

# 运行应用
if __name__ == '__main__':
    # 确保静态文件目录存在
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # 确保模板目录存在
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # 运行应用
    app.run(host='0.0.0.0', port=5000, debug=True)
