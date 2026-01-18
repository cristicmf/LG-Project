#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索工具模块
"""

import re
import math

# 简单的基于相似度的搜索函数（作为备选）
def simple_similarity_search(query, documents, top_k=3):
    """基于 TF-IDF 和余弦相似度进行搜索，优化中文分词和文档标签匹配"""
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
