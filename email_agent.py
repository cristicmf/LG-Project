#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 客户支持邮件代理示例（基于 LangGraph 文档）

# Step 1: 导入必要的库和配置
import os

# 解决 protobuf 版本兼容性问题
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# 导入配置文件
from config import SMTP_CONFIG, IMAP_CONFIG, APP_CONFIG

# 导入监控模块
from monitoring import PerformanceMonitor, log_system_status, get_performance_metrics, get_error_counts, get_function_calls

# 导入日志配置
from logging_config import get_logger

# 获取日志记录器
logger = get_logger()

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
@PerformanceMonitor("read_email")
def read_email(state):
    """读取和解析邮件内容"""
    # 在实际应用中，这里会从邮件服务器或 API 读取邮件
    # 这里我们假设邮件内容已经在状态中
    logger.info("Email content read successfully")
    return state

# 2. 分类意图节点
@PerformanceMonitor("classify_intent")
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
            logger.info("使用 OpenAI API 完成邮件分类")
            
        except Exception as e:
            logger.error("使用 OpenAI API 分类失败，将使用备选逻辑: %s", str(e))
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
            elif "return" in email_content.lower() or "退货" in email_content:
                classification["intent"] = "question"
                classification["urgency"] = "medium"
                classification["topic"] = "return policy"
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
        elif "return" in email_content.lower() or "退货" in email_content:
            classification["intent"] = "question"
            classification["urgency"] = "medium"
            classification["topic"] = "return policy"
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["classification"] = classification
    return new_state

# 尝试导入向量数据库库（作为备选）
vector_db_available = False
try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    # 尝试使用 Milvus Lite
    connections.connect("default", host="localhost", port="19530", secure=False)
    vector_db_available = True
    print("Milvus Lite 向量数据库已安装且可用，可以使用向量搜索进行文档检索")
except ImportError:
    print("未安装 Milvus 向量数据库库，将使用基于相似度的搜索")
except Exception as e:
    print("初始化 Milvus 向量数据库失败:", str(e))

# 初始化向量数据库（作为备选）
collection = None

if vector_db_available:
    try:
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)  # 使用 128 维向量
        ]
        
        # 定义集合 schema
        schema = CollectionSchema(fields, "支持文档向量集合")
        
        # 创建或获取集合
        collection = Collection(name="support_docs", schema=schema)
        
        # 检查集合是否为空，如果为空则添加示例文档
        if collection.num_entities == 0:
            # 从结构化文档库中提取文档内容
            ids = []
            contents = []
            tags_list = []
            
            for i, doc in enumerate(EXAMPLE_DOCUMENTS):
                if isinstance(doc, dict) and 'content' in doc and 'tags' in doc:
                    ids.append(f"doc_{i}")
                    contents.append(doc['content'])
                    tags_list.append(",".join(doc['tags']))
                else:
                    ids.append(f"doc_{i}")
                    contents.append(doc)
                    tags_list.append("")
            
            # 生成简单的嵌入向量（实际应用中应使用真实的嵌入模型）
            import numpy as np
            embeddings = np.random.rand(len(ids), 128).tolist()
            
            # 插入数据
            collection.insert([ids, contents, tags_list, embeddings])
            
            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
                "metric_type": "L2"
            }
            collection.create_index("embedding", index_params)
            
            # 加载集合到内存
            collection.load()
            
            print("示例文档已添加到 Milvus 向量数据库")
    except Exception as e:
        print("初始化 Milvus 向量数据库集合失败:", str(e))
        vector_db_available = False
        collection = None

# 简单的基于相似度的搜索函数（作为备选）
def simple_similarity_search(query, documents, top_k=3):
    """基于 TF-IDF 和余弦相似度进行搜索，优化中文分词和文档标签匹配"""
    import re
    import math
    
    # 检查文档是否为结构化格式
    def is_structured(doc):
        return isinstance(doc, dict) and 'content' in doc and 'tags' in doc
    
    # 获取文档内容
    def get_doc_content(doc):
        return doc['content'] if is_structured(doc) else doc
    
    # 获取文档标签
    def get_doc_tags(doc):
        return doc['tags'] if is_structured(doc) else []
    
    # 预处理文本
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    # 分词函数（支持中英文）
    def tokenize(text):
        # 对于中文，使用字符级分词
        # 对于英文，使用单词级分词
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_words = re.findall(r'[a-zA-Z]+', text)
        return chinese_chars + english_words
    
    # 构建词汇表
    def build_vocab(corpus):
        vocab = set()
        for doc in corpus:
            doc_content = get_doc_content(doc)
            tokens = tokenize(preprocess(doc_content))
            vocab.update(tokens)
        return list(vocab)
    
    # 计算 TF-IDF
    def compute_tfidf(corpus, vocab):
        # 计算 IDF
        idf = {}
        doc_count = len(corpus)
        for term in vocab:
            term_count = 0
            for doc in corpus:
                doc_content = get_doc_content(doc)
                if term in tokenize(preprocess(doc_content)):
                    term_count += 1
            idf[term] = math.log(doc_count / (term_count + 1))
        
        # 计算 TF-IDF
        tfidf = []
        for doc in corpus:
            doc_content = get_doc_content(doc)
            tokens = tokenize(preprocess(doc_content))
            doc_tfidf = {term: 0 for term in vocab}
            for term in tokens:
                if term in vocab:
                    doc_tfidf[term] += 1
            # 归一化 TF
            token_count = len(tokens)
            if token_count > 0:
                for term in vocab:
                    doc_tfidf[term] = (doc_tfidf[term] / token_count) * idf.get(term, 0)
            tfidf.append(doc_tfidf)
        
        return tfidf, idf
    
    # 计算余弦相似度
    def cosine_similarity(vec1, vec2):
        # 计算点积
        dot_product = 0
        for term in vec1:
            dot_product += vec1[term] * vec2.get(term, 0)
        
        # 计算向量长度
        def vector_length(vec):
            return math.sqrt(sum(v**2 for v in vec.values()))
        
        len1 = vector_length(vec1)
        len2 = vector_length(vec2)
        
        # 计算余弦相似度
        if len1 == 0 or len2 == 0:
            return 0
        return dot_product / (len1 * len2)
    
    # 计算标签匹配得分
    def compute_tag_score(query, doc):
        doc_tags = get_doc_tags(doc)
        if not doc_tags:
            return 0
        
        # 提取查询中的关键词
        query_keywords = set(tokenize(preprocess(query)))
        
        # 计算标签匹配数
        matched_tags = 0
        for tag in doc_tags:
            tag_lower = tag.lower()
            if tag_lower in query_keywords or any(keyword in tag_lower for keyword in query_keywords):
                matched_tags += 1
        
        # 计算标签匹配得分
        return matched_tags / len(doc_tags)
    
    # 预处理查询
    query_processed = preprocess(query)
    
    # 构建词汇表
    corpus = documents
    vocab = build_vocab(corpus)
    
    # 计算文档的 TF-IDF
    doc_tfidf, idf = compute_tfidf(corpus, vocab)
    
    # 计算查询的 TF-IDF
    query_tokens = tokenize(query_processed)
    query_tfidf = {term: 0 for term in vocab}
    for term in query_tokens:
        if term in vocab:
            query_tfidf[term] += 1
    # 归一化 TF
    token_count = len(query_tokens)
    if token_count > 0:
        for term in vocab:
            query_tfidf[term] = (query_tfidf[term] / token_count) * idf.get(term, 0)
    
    # 计算每个文档与查询的相似度
    similarities = []
    for i, doc in enumerate(documents):
        # 计算内容相似度
        content_similarity = cosine_similarity(query_tfidf, doc_tfidf[i])
        
        # 计算标签匹配得分
        tag_score = compute_tag_score(query, doc)
        
        # 综合得分（内容相似度占70%，标签匹配占30%）
        total_score = (content_similarity * 0.7) + (tag_score * 0.3)
        
        similarities.append((total_score, doc))
    
    # 按相似度排序并返回前 top_k 个结果
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # 提取结果内容
    results = []
    for _, doc in similarities[:top_k]:
        results.append(get_doc_content(doc))
    
    return results

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

# 3. 文档搜索节点
@PerformanceMonitor("search_docs")
def search_docs(state):
    """搜索相关文档"""
    classification = state["classification"]
    if not classification:
        return state
    
    search_results = []
    topic = classification["topic"]
    email_content = state["email_content"]
    
    # 优化搜索查询构建
    def build_search_query(topic, email_content):
        """根据主题和邮件内容构建优化的搜索查询"""
        import re
        
        # 提取邮件内容中的关键词
        def extract_keywords(text):
            # 移除标点符号
            text = re.sub(r'[^\w\s]', '', text)
            # 分词
            words = text.lower().split()
            # 过滤停用词
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on'])
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            return keywords
        
        # 根据主题构建查询
        topic_keywords = {
            "password reset": ["password", "reset", "忘记密码", "密码重置"],
            "technical issue": ["technical", "error", "crash", "bug", "技术", "错误", "崩溃"],
            "billing issue": ["billing", "charge", "bill", "refund", "账单", "收费", "退款"],
            "feature request": ["feature", "add", "request", "功能", "添加"],
            "return policy": ["return", "policy", "退货", "政策"],
            "technical integration": ["api", "integration", "技术集成"]
        }
        
        # 构建查询
        query_parts = []
        
        # 添加主题关键词
        if topic in topic_keywords:
            query_parts.extend(topic_keywords[topic])
        else:
            query_parts.append(topic)
        
        # 添加邮件内容中的关键词
        email_keywords = extract_keywords(email_content)
        query_parts.extend(email_keywords[:5])  # 只添加前5个关键词，避免查询过长
        
        # 去重并返回查询
        return " ".join(set(query_parts))
    
    # 构建优化的搜索查询
    query = build_search_query(topic, email_content)
    
    # 尝试使用向量数据库进行搜索
    if vector_db_available and collection:
        try:
            # 生成查询向量（实际应用中应使用真实的嵌入模型）
            import numpy as np
            query_embedding = np.random.rand(128).tolist()
            
            # 执行向量搜索
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=3,
                output_fields=["content"]
            )
            
            # 处理搜索结果
            for hits in results:
                for hit in hits:
                    search_results.append(hit.entity.get("content"))
            
            if search_results:
                logger.info("使用 Milvus 向量数据库完成文档搜索")
        except Exception as e:
            logger.error("Milvus 向量数据库搜索失败，将使用基于相似度的搜索: %s", str(e))
            # 继续使用基于相似度的搜索
    
    # 如果向量搜索失败或未启用，使用基于相似度的搜索
    if not search_results:
        try:
            # 执行基于相似度的搜索
            search_results = simple_similarity_search(query, EXAMPLE_DOCUMENTS, top_k=3)
            logger.info("使用基于相似度的搜索完成文档检索")
        except Exception as e:
            logger.error("基于相似度的搜索失败，将使用基于规则的搜索: %s", str(e))
            # 继续使用基于规则的搜索
    
    # 如果相似度搜索失败，使用基于规则的搜索
    if not search_results:
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
@PerformanceMonitor("draft_response")
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
            logger.info("使用 OpenAI API 生成回复")
            
        except Exception as e:
            logger.error("使用 OpenAI API 生成回复失败，将使用备选逻辑: %s", str(e))
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

# 6. 人工审核节点
@PerformanceMonitor("human_review")
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
        logger.info("邮件已通过人工审核并更新")
        return new_state
    logger.warning("人工审核失败：未找到草稿回复")
    return state

# 7. 发送回复节点
@PerformanceMonitor("send_reply")
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
    logger.info("回复已发送到: %s", state['sender_email'])
    return new_state

# 8. 真实邮件发送节点
@PerformanceMonitor("send_real_email")
def send_real_email(state):
    """使用 SMTP 协议发送真实邮件"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    messages = state.get("messages")
    if messages is None:
        messages = []
    
    try:
        # 从配置文件中读取 SMTP 服务器配置
        smtp_server = SMTP_CONFIG["server"]
        smtp_port = SMTP_CONFIG["port"]
        smtp_username = SMTP_CONFIG["username"]
        smtp_password = SMTP_CONFIG["password"]
        sender_email = SMTP_CONFIG["sender_email"]
        recipient_email = state['sender_email']
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Re: Your Support Request"
        
        # 添加邮件正文
        body = state.get("draft_response", "")
        msg.attach(MIMEText(body, 'plain'))
        
        # 连接到 SMTP 服务器并发送邮件
        # 注意：实际发送需要正确的服务器配置
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            
            messages.append(f"Real email sent to {recipient_email}")
            logger.info(f"Email sent to {recipient_email}")
        except Exception as e:
            # 如果实际发送失败，使用模拟发送
            messages.append(f"Real email sent to {recipient_email} (simulated - SMTP connection failed: {str(e)})")
            logger.warning(f"Simulated email sent to {recipient_email} (SMTP connection failed: {str(e)})")
        
    except Exception as e:
        error_message = f"Failed to send email: {str(e)}"
        messages.append(error_message)
        logger.error(error_message)
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["messages"] = messages
    return new_state

# 9. 邮件接收节点
@PerformanceMonitor("receive_email")
def receive_email(state):
    """使用 IMAP 协议接收邮件"""
    import imaplib
    import email
    from email.header import decode_header
    
    messages = state.get("messages")
    if messages is None:
        messages = []
    
    try:
        # 从配置文件中读取 IMAP 服务器配置
        imap_server = IMAP_CONFIG["server"]
        imap_username = IMAP_CONFIG["username"]
        imap_password = IMAP_CONFIG["password"]
        
        # 连接到 IMAP 服务器并选择收件箱
        # 注意：实际接收需要正确的服务器配置
        try:
            with imaplib.IMAP4_SSL(imap_server) as imap:
                imap.login(imap_username, imap_password)
                status, messages_count = imap.select("INBOX")
                
                # 搜索未读邮件
                status, messages_ids = imap.search(None, "UNSEEN")
                
                # 处理每封邮件
                for msg_id in messages_ids[0].split():
                    status, msg_data = imap.fetch(msg_id, "(RFC822)")
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            # 解析邮件
                            subject, encoding = decode_header(msg["Subject"])[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding if encoding else "utf-8")
                            
                            from_email = msg.get("From")
                            
                            # 获取邮件正文
                            if msg.is_multipart():
                                for part in msg.walk():
                                    content_type = part.get_content_type()
                                    content_disposition = str(part.get("Content-Disposition"))
                                    if content_type == "text/plain" and "attachment" not in content_disposition:
                                        body = part.get_payload(decode=True).decode()
                                        break
                            else:
                                body = msg.get_payload(decode=True).decode()
                            
                            # 处理邮件
                            logger.info(f"Received email from {from_email} with subject: {subject}")
                            messages.append(f"Email received from {from_email} with subject: {subject}")
        except Exception as e:
            # 如果实际接收失败，使用模拟接收
            messages.append(f"Email received from {state['sender_email']} (simulated - IMAP connection failed: {str(e)})")
            logger.warning(f"Simulated email received from {state['sender_email']} (IMAP connection failed: {str(e)})")
        
    except Exception as e:
        error_message = f"Failed to receive email: {str(e)}"
        messages.append(error_message)
        logger.error(error_message)
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["messages"] = messages
    return new_state

# 10. 邮件归档节点
@PerformanceMonitor("archive_email")
def archive_email(state):
    """归档邮件到历史记录"""
    import json
    import os
    from datetime import datetime
    
    messages = state.get("messages")
    if messages is None:
        messages = []
    
    try:
        # 从配置文件中读取归档目录配置
        archive_dir = APP_CONFIG.get("email_archive_dir", "email_archive")
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        
        # 准备归档数据
        archive_data = {
            "email_id": state['email_id'],
            "sender_email": state['sender_email'],
            "email_content": state['email_content'],
            "classification": state.get("classification", {}),
            "draft_response": state.get("draft_response", ""),
            "customer_history": state.get("customer_history", {}),
            "archived_at": datetime.now().isoformat()
        }
        
        # 生成归档文件名
        archive_filename = os.path.join(archive_dir, f"email_{state['email_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # 保存归档数据
        with open(archive_filename, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, ensure_ascii=False, indent=2)
        
        messages.append(f"Email archived: {state['email_id']} - saved to {archive_filename}")
        logger.info(f"Email archived: {state['email_id']} - saved to {archive_filename}")
        
    except Exception as e:
        error_message = f"Failed to archive email: {str(e)}"
        messages.append(error_message)
        logger.error(error_message)
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["messages"] = messages
    return new_state

# 11. 客户历史记录查询节点
@PerformanceMonitor("query_customer_history")
def query_customer_history(state):
    """查询客户历史记录"""
    # 在实际应用中，这里会从 CRM 系统查询客户历史记录
    # 这里我们只是模拟查询结果
    customer_history = {
        "customer_id": "12345",
        "name": "John Doe",
        "email": state['sender_email'],
        "previous_issues": ["password reset", "technical issue"],
        "membership_level": "premium",
        "last_contact": "2023-07-01",
        "support_tier": "enterprise"
    }
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["customer_history"] = customer_history
    logger.info(f"Customer history queried for: {state['sender_email']}")
    return new_state

# 12. 邮件优先级处理节点
@PerformanceMonitor("process_priority")
def process_priority(state):
    """处理邮件优先级"""
    classification = state.get("classification", {})
    customer_history = state.get("customer_history", {})
    
    # 基于分类和客户历史设置优先级
    priority = classification.get("urgency", "medium")
    
    # 为高级客户提高优先级
    if customer_history.get("membership_level") == "premium":
        if priority == "medium":
            priority = "high"
        elif priority == "low":
            priority = "medium"
    
    # 创建新状态并返回
    new_state = state.copy()
    if "classification" in new_state and new_state["classification"]:
        new_state["classification"]["priority"] = priority
    
    # 获取 messages 键
    messages = new_state.get("messages", None)
    if messages is None:
        messages = []
    messages.append(f"Priority set to: {priority}")
    # 设置 messages 键
    new_state["messages"] = messages
    
    logger.info(f"Priority set to: {priority} for email from {state.get('sender_email')}")
    return new_state

# 13. 邮件转发节点
@PerformanceMonitor("forward_email")
def forward_email(state):
    """转发邮件到相关部门"""
    classification = state.get("classification", {})
    topic = classification.get("topic", "")
    
    # 基于主题确定转发目标
    forward_targets = {
        "password reset": "support@example.com",
        "technical issue": "tech-support@example.com",
        "billing issue": "billing@example.com",
        "return policy": "returns@example.com",
        "feature request": "product@example.com"
    }
    
    target = forward_targets.get(topic, "support@example.com")
    
    # 创建新状态并返回
    new_state = state.copy()
    messages = new_state.get("messages", [])
    messages.append(f"Email forwarded to: {target}")
    new_state["messages"] = messages
    
    logger.info(f"Email forwarded to: {target}")
    
    return new_state

# 14. 邮件模板选择节点
@PerformanceMonitor("select_email_template")
def select_email_template(state):
    """根据邮件类型选择合适的模板"""
    classification = state.get("classification", {})
    intent = classification.get("intent", "question")
    
    # 模板库
    templates = {
        "question": "Thank you for your question. We're here to help you with any inquiries you may have.",
        "bug": "We're sorry to hear you're experiencing an issue. Our team is dedicated to resolving technical problems quickly.",
        "billing": "Thank you for bringing this billing matter to our attention. We'll work to resolve it promptly.",
        "feature": "Thank you for your feature request. We value your feedback and are always looking to improve our services.",
        "complex": "Thank you for reaching out about this complex issue. One of our senior specialists will assist you shortly."
    }
    
    template = templates.get(intent, templates["question"])
    
    # 创建新状态并返回
    new_state = state.copy()
    new_state["email_template"] = template
    
    messages = new_state.get("messages", [])
    messages.append(f"Email template selected: {intent}")
    new_state["messages"] = messages
    
    logger.info(f"Email template selected: {intent}")
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
@PerformanceMonitor("build_email_agent")
def build_email_agent():
    """构建邮件代理图"""
    # 创建图
    graph = SimpleGraph()
    
    # 添加节点
    graph.add_node("read_email", read_email)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("query_customer_history", query_customer_history)
    graph.add_node("process_priority", process_priority)
    graph.add_node("search_docs", search_docs)
    graph.add_node("select_email_template", select_email_template)
    graph.add_node("draft_response", draft_response)
    graph.add_node("human_review", human_review)
    graph.add_node("send_reply", send_reply)
    graph.add_node("send_real_email", send_real_email)
    graph.add_node("forward_email", forward_email)
    graph.add_node("archive_email", archive_email)
    
    # 添加边
    graph.add_edge("read_email", "classify_intent")
    graph.add_edge("classify_intent", "query_customer_history")
    graph.add_edge("query_customer_history", "process_priority")
    graph.add_edge("process_priority", "search_docs")
    graph.add_edge("search_docs", "select_email_template")
    graph.add_edge("select_email_template", "draft_response")
    
    # 添加条件边
    graph.add_conditional_edge(
        "draft_response",
        check_escalation,
        {"send_reply": "send_reply", "human_review": "human_review"}
    )
    
    graph.add_edge("human_review", "forward_email")
    graph.add_edge("send_reply", "forward_email")
    graph.add_edge("forward_email", "send_real_email")
    graph.add_edge("send_real_email", "archive_email")
    
    # 设置入口点
    graph.set_entry_point("read_email")
    
    # 编译图
    compiled_graph = graph.compile()
    logger.info("Email agent built successfully")
    return compiled_graph

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
    
    # 导入测试用例
    from test_cases import test_email_1, test_email_2, test_email_3, test_email_4
    
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
    
    print("\nTest Case 4: Return Issue")
    result4 = agent.invoke(test_email_4)
    print("Classification:", result4.classification)
    print("Draft Response:", result4.draft_response)
