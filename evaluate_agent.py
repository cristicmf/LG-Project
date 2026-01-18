#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估邮件代理模块
用于评估邮件处理代理的性能和准确性
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 导入日志配置
from logging_config import get_logger

# 获取日志记录器
logger = get_logger()

# 导入监控模块
from monitoring import PerformanceMonitor

# 导入邮件代理
from src.app.email_agent import EmailAgent


class AgentEvaluator:
    """
    邮件代理评估器
    用于评估邮件处理代理的性能和准确性
    """
    
    def __init__(self):
        """初始化评估器"""
        self.agent = EmailAgent()
        self.evaluation_results = []
        self.evaluation_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    @PerformanceMonitor("evaluate_agent")
    def evaluate(self, test_cases: List[Dict], evaluate_accuracy: bool = False) -> Dict:
        """
        评估邮件代理
        
        Args:
            test_cases: 测试用例列表
            evaluate_accuracy: 是否评估准确性
            
        Returns:
            评估结果
        """
        start_time = time.time()
        total_response_time = 0
        correct_predictions = 0
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"开始评估测试用例 {i+1}/{len(test_cases)}")
            
            # 记录单个测试用例的开始时间
            test_start_time = time.time()
            
            try:
                # 执行邮件代理
                email_content = test_case.get("email_content", "")
                result = self.agent.process_email(email_content)
                
                # 记录单个测试用例的执行时间
                test_response_time = time.time() - test_start_time
                total_response_time += test_response_time
                
                # 评估准确性
                if evaluate_accuracy:
                    expected_classification = test_case.get("expected_classification", {})
                    actual_classification = result.get("classification", {})
                    
                    if self._evaluate_accuracy(expected_classification, actual_classification):
                        correct_predictions += 1
                
                # 记录成功的测试用例
                self.evaluation_results.append({
                    "test_case": test_case,
                    "result": result,
                    "response_time": test_response_time,
                    "status": "success"
                })
                
                self.evaluation_metrics["successful_requests"] += 1
                logger.info(f"测试用例 {i+1} 执行成功，响应时间: {test_response_time:.2f} 秒")
                
            except Exception as e:
                # 记录失败的测试用例
                self.evaluation_results.append({
                    "test_case": test_case,
                    "error": str(e),
                    "response_time": time.time() - test_start_time,
                    "status": "failed"
                })
                
                self.evaluation_metrics["failed_requests"] += 1
                logger.error(f"测试用例 {i+1} 执行失败: {str(e)}")
            
            self.evaluation_metrics["total_requests"] += 1
        
        # 计算评估指标
        total_time = time.time() - start_time
        self.evaluation_metrics["average_response_time"] = total_response_time / len(test_cases) if test_cases else 0
        
        if evaluate_accuracy and test_cases:
            self.evaluation_metrics["accuracy"] = correct_predictions / len(test_cases)
            # 计算 precision, recall, f1_score
            self._calculate_precision_recall_f1(test_cases)
        
        # 保存评估结果
        self._save_evaluation_results()
        
        logger.info(f"评估完成，总执行时间: {total_time:.2f} 秒，平均响应时间: {self.evaluation_metrics['average_response_time']:.2f} 秒")
        logger.info(f"成功: {self.evaluation_metrics['successful_requests']}, 失败: {self.evaluation_metrics['failed_requests']}")
        if evaluate_accuracy:
            logger.info(f"准确率: {self.evaluation_metrics['accuracy']:.2f}, F1 分数: {self.evaluation_metrics['f1_score']:.2f}")
        
        return self.evaluation_metrics
    
    def _evaluate_accuracy(self, expected: Dict, actual: Dict) -> bool:
        """
        评估分类准确性
        
        Args:
            expected: 预期分类结果
            actual: 实际分类结果
            
        Returns:
            是否准确
        """
        # 比较意图、紧急程度和主题
        if expected.get("intent") != actual.get("intent"):
            return False
        if expected.get("urgency") != actual.get("urgency"):
            return False
        if expected.get("topic") != actual.get("topic"):
            return False
        return True
    
    def _calculate_precision_recall_f1(self, test_cases: List[Dict]):
        """
        计算精确率、召回率和 F1 分数
        
        Args:
            test_cases: 测试用例列表
        """
        # 计算混淆矩阵
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i, test_case in enumerate(test_cases):
            result = self.evaluation_results[i]
            if result["status"] == "success":
                expected = test_case.get("expected_classification", {})
                actual = result["result"].get("classification", {})
                
                if self._evaluate_accuracy(expected, actual):
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                false_negatives += 1
        
        # 计算精确率
        if true_positives + false_positives > 0:
            self.evaluation_metrics["precision"] = true_positives / (true_positives + false_positives)
        else:
            self.evaluation_metrics["precision"] = 0.0
        
        # 计算召回率
        if true_positives + false_negatives > 0:
            self.evaluation_metrics["recall"] = true_positives / (true_positives + false_negatives)
        else:
            self.evaluation_metrics["recall"] = 0.0
        
        # 计算 F1 分数
        if self.evaluation_metrics["precision"] + self.evaluation_metrics["recall"] > 0:
            self.evaluation_metrics["f1_score"] = 2 * (
                self.evaluation_metrics["precision"] * self.evaluation_metrics["recall"]
            ) / (
                self.evaluation_metrics["precision"] + self.evaluation_metrics["recall"]
            )
        else:
            self.evaluation_metrics["f1_score"] = 0.0
    
    def _save_evaluation_results(self):
        """
        保存评估结果到文件
        """
        # 创建评估结果目录
        evaluation_dir = "evaluation_results"
        if not os.path.exists(evaluation_dir):
            os.makedirs(evaluation_dir)
        
        # 生成评估结果文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_file = os.path.join(evaluation_dir, f"evaluation_{timestamp}.json")
        
        # 保存评估结果
        evaluation_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.evaluation_metrics,
            "results": self.evaluation_results
        }
        
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存到: {evaluation_file}")
    
    def get_evaluation_metrics(self) -> Dict:
        """
        获取评估指标
        
        Returns:
            评估指标
        """
        return self.evaluation_metrics
    
    def get_evaluation_results(self) -> List[Dict]:
        """
        获取评估结果
        
        Returns:
            评估结果
        """
        return self.evaluation_results


# 示例测试用例
def get_sample_test_cases():
    """
    获取示例测试用例
    
    Returns:
        示例测试用例列表
    """
    return [
        {
            "email_id": "1",
            "sender_email": "user1@example.com",
            "email_content": "I forgot my password and can't log in. Please help me reset it.",
            "expected_classification": {
                "intent": "question",
                "urgency": "medium",
                "topic": "password reset"
            }
        },
        {
            "email_id": "2",
            "sender_email": "user2@example.com",
            "email_content": "The application crashes every time I try to open it. Error code: 504.",
            "expected_classification": {
                "intent": "bug",
                "urgency": "high",
                "topic": "technical issue"
            }
        },
        {
            "email_id": "3",
            "sender_email": "user3@example.com",
            "email_content": "I was charged twice for my subscription. Please refund the extra amount.",
            "expected_classification": {
                "intent": "billing",
                "urgency": "high",
                "topic": "billing issue"
            }
        },
        {
            "email_id": "4",
            "sender_email": "user4@example.com",
            "email_content": "It would be great if you could add dark mode to the application.",
            "expected_classification": {
                "intent": "feature",
                "urgency": "low",
                "topic": "feature request"
            }
        },
        {
            "email_id": "5",
            "sender_email": "user5@example.com",
            "email_content": "I need help integrating your API with my system. Can you provide detailed documentation?",
            "expected_classification": {
                "intent": "complex",
                "urgency": "medium",
                "topic": "technical integration"
            }
        }
    ]


# 主函数
if __name__ == "__main__":
    # 创建评估器
    evaluator = AgentEvaluator()
    
    # 获取示例测试用例
    test_cases = get_sample_test_cases()
    
    # 评估邮件代理
    logger.info("开始评估邮件处理代理...")
    metrics = evaluator.evaluate(test_cases, evaluate_accuracy=True)
    
    # 打印评估结果
    logger.info("评估结果:")
    logger.info(f"总请求数: {metrics['total_requests']}")
    logger.info(f"成功请求数: {metrics['successful_requests']}")
    logger.info(f"失败请求数: {metrics['failed_requests']}")
    logger.info(f"平均响应时间: {metrics['average_response_time']:.2f} 秒")
    logger.info(f"准确率: {metrics['accuracy']:.2f}")
    logger.info(f"精确率: {metrics['precision']:.2f}")
    logger.info(f"召回率: {metrics['recall']:.2f}")
    logger.info(f"F1 分数: {metrics['f1_score']:.2f}")
