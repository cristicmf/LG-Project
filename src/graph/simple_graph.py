#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单图实现模块
模拟 LangGraph 的核心功能
"""

class SimpleGraph:
    """简单图类，模拟 LangGraph 的核心功能"""
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
    
    def add_node(self, name, func):
        """添加节点"""
        self.nodes[name] = func
    
    def add_edge(self, start, end):
        """添加边"""
        self.edges[start] = end
    
    def add_conditional_edge(self, start, condition_func, mapping):
        """添加条件边"""
        self.edges[start] = (condition_func, mapping)
    
    def set_entry_point(self, name):
        """设置入口点"""
        self.entry_point = name
    
    def compile(self):
        """编译图"""
        return self
    
    def invoke(self, initial_state):
        """执行图"""
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
