#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索文档节点
"""

import os
import sys
import re
import math

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from monitoring import PerformanceMonitor
from logging_config import get_logger
from src.tools.search_tool import simple_similarity_search
from src.data.documents import get_documents

logger = get_logger()

# 尝试导入向量数据库库（作为备选）
vector_db_available = False
try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    # 尝试使用 Milvus Lite
    connections.connect("default", host="localhost", port="19530", secure=False)
    vector_db_available = True
    logger.info("Milvus Lite 向量数据库已安装且可用，可以使用向量搜索进行文档检索")
except ImportError:
    logger.info("未安装 Milvus 向量数据库库，将使用基于相似度的搜索")
except Exception as e:
    logger.warning("初始化 Milvus 向量数据库失败: %s", str(e))

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
            
            documents = get_documents()
            for i, doc in enumerate(documents):
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
            
            logger.info("示例文档已添加到 Milvus 向量数据库")
    except Exception as e:
        logger.warning("初始化 Milvus 向量数据库集合失败: %s", str(e))
        vector_db_available = False
        collection = None

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
            documents = get_documents()
            search_results = simple_similarity_search(query, documents, top_k=3)
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
