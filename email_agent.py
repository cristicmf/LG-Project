#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 客户支持邮件代理示例（基于 LangGraph 文档）

# Step 1: 导入必要的库和配置
import os

# 直接从环境变量中读取配置（不使用 dotenv 以避免环境问题）

# 尝试导入 OpenAI API
openai_available = False
try:
    from openai import OpenAI
    # 检查是否设置了 API 密钥
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        openai_available = True
        print("OpenAI API 已配置，可以使用真实的 LLM 进行邮件分类和回复生成")
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
                            print("从 .env 文件读取到 OpenAI API 密钥，可以使用真实的 LLM 进行邮件分类和回复生成")
                            break
        except Exception as e:
            print("读取 .env 文件失败:", str(e))
        
        if not openai_available:
            print("未设置 OPENAI_API_KEY 环境变量，将使用模拟 LLM")
except ImportError:
    print("未安装 openai 库，将使用模拟 LLM")

# Step 2: 定义状态结构

# 定义邮件代理状态（简化版，兼容 Python 3.8）
class EmailAgentState:
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

# Step 3: 定义节点函数

# 1. 读取邮件节点
def read_email(state):
    """读取和解析邮件内容"""
    # 在实际应用中，这里会从邮件服务器或 API 读取邮件
    # 这里我们假设邮件内容已经在状态中
    return state

# 2. 分类意图节点
def classify_intent(state):
    """使用 LLM 对邮件进行分类"""
    email_content = state["email_content"]
    
    # 简单的分类逻辑（作为备选）
    classification = {
        "intent": "question",
        "urgency": "medium",
        "topic": "general",
        "summary": "Customer support request"
    }
    
    # 尝试使用 OpenAI API 进行分类
    if openai_available:
        try:
            # 构建分类提示（使用普通字符串连接避免 f-string 转义问题）
            prompt = "请对以下客户支持邮件进行分类：\n\n"
            prompt += "邮件内容：\n"
            prompt += email_content + "\n\n"
            prompt += "请按照以下格式返回分类结果（使用 JSON 格式）：\n"
            prompt += '{"intent": "question|bug|billing|feature|complex", "urgency": "low|medium|high|critical", "topic": "主题描述", "summary": "邮件摘要"}\n\n'
            prompt += "分类说明：\n"
            prompt += "- intent: 邮件意图\n"
            prompt += "  - question: 一般问题（如密码重置、使用说明等）\n"
            prompt += "  - bug: 技术问题（如崩溃、错误等）\n"
            prompt += "  - billing: 账单问题（如收费、退款等）\n"
            prompt += "  - feature: 功能请求（如添加新功能）\n"
            prompt += "  - complex: 复杂问题（如 API 集成、系统配置等）\n"
            prompt += "- urgency: 紧急程度\n"
            prompt += "  - low: 低优先级\n"
            prompt += "  - medium: 中优先级\n"
            prompt += "  - high: 高优先级\n"
            prompt += "  - critical: 紧急优先级\n"
            prompt += "- topic: 邮件主题（简短描述）\n"
            prompt += "- summary: 邮件摘要（简要总结邮件内容）"
            
            # 调用 OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的客户支持邮件分类助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # 解析 API 响应
            import json
            classification = json.loads(response.choices[0].message.content)
            print("使用 OpenAI API 完成邮件分类")
            
        except Exception as e:
            print("使用 OpenAI API 分类失败，将使用备选逻辑:", str(e))
            # 执行备选逻辑
            if "password" in email_content.lower():
                classification["intent"] = "question"
                classification["topic"] = "password reset"
            elif "crash" in email_content.lower() or "error" in email_content.lower():
                classification["intent"] = "bug"
                classification["urgency"] = "high"
                classification["topic"] = "technical issue"
            elif "charge" in email_content.lower() or "bill" in email_content.lower():
                classification["intent"] = "billing"
                classification["urgency"] = "high"
                classification["topic"] = "billing issue"
            elif "feature" in email_content.lower() or "add" in email_content.lower():
                classification["intent"] = "feature"
                classification["urgency"] = "low"
                classification["topic"] = "feature request"
            elif "api" in email_content.lower() or "integration" in email_content.lower():
                classification["intent"] = "complex"
                classification["urgency"] = "medium"
                classification["topic"] = "technical integration"
    else:
        # 使用备选逻辑进行分类
        if "password" in email_content.lower():
            classification["intent"] = "question"
            classification["topic"] = "password reset"
        elif "crash" in email_content.lower() or "error" in email_content.lower():
            classification["intent"] = "bug"
            classification["urgency"] = "high"
            classification["topic"] = "technical issue"
        elif "charge" in email_content.lower() or "bill" in email_content.lower():
            classification["intent"] = "billing"
            classification["urgency"] = "high"
            classification["topic"] = "billing issue"
        elif "feature" in email_content.lower() or "add" in email_content.lower():
            classification["intent"] = "feature"
            classification["urgency"] = "low"
            classification["topic"] = "feature request"
        elif "api" in email_content.lower() or "integration" in email_content.lower():
            classification["intent"] = "complex"
            classification["urgency"] = "medium"
            classification["topic"] = "technical integration"
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["classification"] = classification
    return new_state

# 3. 文档搜索节点
def search_docs(state):
    """搜索相关文档"""
    classification = state["classification"]
    if not classification:
        return state
    
    # 简单的搜索结果（实际应用中会使用向量数据库或搜索引擎）
    search_results = []
    topic = classification["topic"]
    
    if "password" in topic.lower():
        search_results = [
            "如何重置密码：1. 访问登录页面 2. 点击'忘记密码' 3. 按照邮件指示操作",
            "密码重置常见问题：密码重置链接有效期为24小时"
        ]
    elif "technical" in topic.lower():
        search_results = [
            "常见技术问题排查：1. 清除浏览器缓存 2. 尝试使用不同浏览器 3. 检查网络连接",
            "错误代码参考：504 错误通常表示服务器超时"
        ]
    elif "billing" in topic.lower():
        search_results = [
            "账单问题处理：如果您被重复收费，请联系客服提供交易ID",
            "退款政策：符合条件的退款将在3-5个工作日内处理"
        ]
    elif "feature" in topic.lower():
        search_results = [
            "功能请求流程：我们会定期评估用户提出的功能请求",
            "即将推出的功能：我们计划在下个季度添加暗黑模式"
        ]
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["search_results"] = search_results
    return new_state

# 4. 起草回复节点
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
            print("使用 OpenAI API 生成回复")
            
        except Exception as e:
            print("使用 OpenAI API 生成回复失败，将使用备选逻辑:", str(e))
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

# 5. 检查是否需要人工审核节点
def check_escalation(state):
    """决定是否需要人工审核"""
    classification = state["classification"]
    if not classification:
        return "human_review"
    
    # 基于紧急程度和意图决定是否需要人工审核
    urgency = classification["urgency"]
    intent = classification["intent"]
    
    if urgency in ["high", "critical"] or intent == "complex":
        return "human_review"
    else:
        return "send_reply"

# 6. 人工审核节点
def human_review(state):
    """人工审核和处理"""
    # 在实际应用中，这里会将邮件发送给人工代理进行审核
    # 这里我们假设人工代理已经审核并更新了回复
    draft_response = state["draft_response"]
    if draft_response:
        # 模拟人工审核后的回复
        reviewed_response = draft_response.replace("Best regards", "Kind regards")
        reviewed_response += "\n\nP.S. This response has been reviewed by a human agent."
        # 创建新状态并返回
        new_state = state.copy()
        new_state["draft_response"] = reviewed_response
        return new_state
    return state

# 7. 发送回复节点
def send_reply(state):
    """发送回复"""
    # 在实际应用中，这里会通过邮件服务器发送回复
    # 这里我们只是记录发送状态
    messages = state.get("messages")
    if messages is None:
        messages = []
    messages.append("Reply sent to " + state['sender_email'])
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["messages"] = messages
    return new_state

# Step 4: 连接节点并构建图
# 注意：由于 Python 版本限制，这里使用模拟实现
# 在 Python 3.9+ 环境中，可以使用完整的 LangGraph 实现

# 模拟 LangGraph 的简单实现
class SimpleGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
    
    def add_node(self, name, func):
        self.nodes[name] = func
    
    def add_edge(self, start, end):
        self.edges[start] = end
    
    def add_conditional_edge(self, start, condition_func, mapping):
        self.edges[start] = (condition_func, mapping)
    
    def set_entry_point(self, name):
        self.entry_point = name
    
    def compile(self):
        return self
    
    def invoke(self, initial_state):
        current_state = initial_state
        current_node = self.entry_point
        
        while current_node:
            # 执行当前节点
            if current_node in self.nodes:
                current_state = self.nodes[current_node](current_state)
            
            # 确定下一个节点
            if current_node in self.edges:
                next_node = self.edges[current_node]
                
                # 处理条件边
                if isinstance(next_node, tuple):
                    condition_func, mapping = next_node
                    condition_result = condition_func(current_state)
                    current_node = mapping.get(condition_result)
                else:
                    current_node = next_node
            else:
                current_node = None
        
        return current_state

# Step 5: 构建和编译图
def build_email_agent():
    """构建邮件代理图"""
    # 创建图
    graph = SimpleGraph()
    
    # 添加节点
    graph.add_node("read_email", read_email)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("search_docs", search_docs)
    graph.add_node("draft_response", draft_response)
    graph.add_node("human_review", human_review)
    graph.add_node("send_reply", send_reply)
    
    # 添加边
    graph.add_edge("read_email", "classify_intent")
    graph.add_edge("classify_intent", "search_docs")
    graph.add_edge("search_docs", "draft_response")
    
    # 添加条件边
    graph.add_conditional_edge(
        "draft_response",
        check_escalation,
        {"send_reply": "send_reply", "human_review": "human_review"}
    )
    
    graph.add_edge("human_review", "send_reply")
    
    # 设置入口点
    graph.set_entry_point("read_email")
    
    # 编译图
    return graph.compile()

# Step 6: 测试邮件代理
if __name__ == "__main__":
    # 解析命令行参数
    import sys
    use_real_llm = False
    if len(sys.argv) > 1 and sys.argv[1] == "1":
        use_real_llm = True
    
    # 如果不使用真实 LLM，强制设置 openai_available 为 False
    if not use_real_llm:
        openai_available = False
        print("使用虚拟 LLM 运行")
    
    # 创建邮件代理
    agent = build_email_agent()
    
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
    
    # 运行测试
    print("=== 测试邮件代理 ===")
    
    print("\nTest Case 1: Password Reset Request")
    result1 = agent.invoke(test_email_1)
    print("Classification:", result1.classification)
    print("Draft Response:", result1.draft_response)
    
    print("\nTest Case 2: Technical Issue")
    result2 = agent.invoke(test_email_2)
    print("Classification:", result2.classification)
    print("Draft Response:", result2.draft_response)
    
    print("\nTest Case 3: Billing Issue")
    result3 = agent.invoke(test_email_3)
    print("Classification:", result3.classification)
    print("Draft Response:", result3.draft_response)
