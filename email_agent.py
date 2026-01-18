# 客户支持邮件代理示例（基于 LangGraph 文档）

# Step 1: 定义状态结构
from typing import TypedDict, Literal, Optional, List, Dict

# 定义邮件分类结构
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

# 定义邮件代理状态
class EmailAgentState(TypedDict):
    # 原始邮件数据
    email_content: str
    sender_email: str
    email_id: str

    # 分类结果
    classification: Optional[EmailClassification]

    # 搜索/API 结果
    search_results: Optional[List[str]]  # 原始文档块列表
    customer_history: Optional[Dict]  # 来自 CRM 的原始客户数据

    # 生成的内容
    draft_response: Optional[str]
    messages: Optional[List[str]]

# Step 2: 定义节点函数

# 1. 读取邮件节点
def read_email(state: EmailAgentState) -> EmailAgentState:
    """读取和解析邮件内容"""
    # 在实际应用中，这里会从邮件服务器或 API 读取邮件
    # 这里我们假设邮件内容已经在状态中
    return state

# 2. 分类意图节点
def classify_intent(state: EmailAgentState) -> EmailAgentState:
    """使用 LLM 对邮件进行分类"""
    email_content = state["email_content"]
    
    # 简单的分类逻辑（实际应用中会使用 LLM）
    classification = {
        "intent": "question",
        "urgency": "medium",
        "topic": "general",
        "summary": "Customer support request"
    }
    
    # 根据邮件内容进行简单分类
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
    
    return {
        **state,
        "classification": classification
    }

# 3. 文档搜索节点
def search_docs(state: EmailAgentState) -> EmailAgentState:
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
    
    return {
        **state,
        "search_results": search_results
    }

# 4. 起草回复节点
def draft_response(state: EmailAgentState) -> EmailAgentState:
    """生成适当的回复"""
    classification = state["classification"]
    search_results = state["search_results"]
    email_content = state["email_content"]
    
    if not classification:
        return state
    
    # 基于分类和搜索结果起草回复
    draft = ""
    intent = classification["intent"]
    urgency = classification["urgency"]
    
    # 开头
    draft += f"Dear Customer,\n\n"
    
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
    
    return {
        **state,
        "draft_response": draft
    }

# 5. 检查是否需要人工审核节点
def check_escalation(state: EmailAgentState) -> Literal["send_reply", "human_review"]:
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
def human_review(state: EmailAgentState) -> EmailAgentState:
    """人工审核和处理"""
    # 在实际应用中，这里会将邮件发送给人工代理进行审核
    # 这里我们假设人工代理已经审核并更新了回复
    draft_response = state["draft_response"]
    if draft_response:
        # 模拟人工审核后的回复
        reviewed_response = draft_response.replace("Best regards", "Kind regards")
        reviewed_response += "\n\nP.S. This response has been reviewed by a human agent."
        return {
            **state,
            "draft_response": reviewed_response
        }
    return state

# 7. 发送回复节点
def send_reply(state: EmailAgentState) -> EmailAgentState:
    """发送回复"""
    # 在实际应用中，这里会通过邮件服务器发送回复
    # 这里我们只是记录发送状态
    messages = state.get("messages", [])
    messages.append(f"Reply sent to {state['sender_email']}")
    
    return {
        **state,
        "messages": messages
    }

# Step 3: 连接节点并构建图
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

# Step 4: 构建和编译图
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

# Step 5: 测试邮件代理
if __name__ == "__main__":
    # 创建邮件代理
    agent = build_email_agent()
    
    # 测试用例 1: 密码重置问题
    test_email_1 = {
        "email_content": "Hello, I forgot my password and need to reset it. Can you help?",
        "sender_email": "user1@example.com",
        "email_id": "123"
    }
    
    # 测试用例 2: 技术问题
    test_email_2 = {
        "email_content": "The export feature crashes when I select PDF format. Please help!",
        "sender_email": "user2@example.com",
        "email_id": "456"
    }
    
    # 测试用例 3: 账单问题
    test_email_3 = {
        "email_content": "I was charged twice for my subscription. Please refund the extra amount.",
        "sender_email": "user3@example.com",
        "email_id": "789"
    }
    
    # 运行测试
    print("=== 测试邮件代理 ===")
    
    print("\nTest Case 1: Password Reset Request")
    result1 = agent.invoke(test_email_1)
    print(f"Classification: {result1.get('classification')}")
    print(f"Draft Response: {result1.get('draft_response')}")
    
    print("\nTest Case 2: Technical Issue")
    result2 = agent.invoke(test_email_2)
    print(f"Classification: {result2.get('classification')}")
    print(f"Draft Response: {result2.get('draft_response')}")
    
    print("\nTest Case 3: Billing Issue")
    result3 = agent.invoke(test_email_3)
    print(f"Classification: {result3.get('classification')}")
    print(f"Draft Response: {result3.get('draft_response')}")
