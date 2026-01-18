# 监控模块
import time
import logging
from datetime import datetime
from collections import defaultdict

# 性能指标存储
performance_metrics = defaultdict(list)

# 错误计数
error_counts = defaultdict(int)

# 功能调用计数
function_calls = defaultdict(int)

class PerformanceMonitor:
    """性能监控装饰器"""
    def __init__(self, function_name):
        self.function_name = function_name
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_calls[self.function_name] += 1
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 记录性能指标
                performance_metrics[self.function_name].append({
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": execution_time,
                    "success": True
                })
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                error_counts[self.function_name] += 1
                
                # 记录错误
                performance_metrics[self.function_name].append({
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": execution_time,
                    "success": False,
                    "error": str(e)
                })
                
                raise
        
        return wrapper

def get_performance_metrics():
    """获取性能指标"""
    return dict(performance_metrics)

def get_error_counts():
    """获取错误计数"""
    return dict(error_counts)

def get_function_calls():
    """获取功能调用计数"""
    return dict(function_calls)

def reset_metrics():
    """重置监控指标"""
    global performance_metrics, error_counts, function_calls
    performance_metrics = defaultdict(list)
    error_counts = defaultdict(int)
    function_calls = defaultdict(int)

def log_system_status():
    """记录系统状态"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "performance_metrics": get_performance_metrics(),
        "error_counts": get_error_counts(),
        "function_calls": get_function_calls()
    }
    
    logging.info("System status: %s", status)
    return status
