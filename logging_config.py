# 日志配置模块
import logging
import os
import json
from datetime import datetime

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# 日志目录
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 日志文件路径
LOG_FILE = os.path.join(LOG_DIR, f"email_agent_{datetime.now().strftime('%Y%m%d')}.log")

# 结构化日志格式化器
class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    def format(self, record):
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)

def setup_logging(log_level="INFO"):
    """设置日志配置"""
    # 获取日志级别
    level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    
    # 创建日志记录器
    logger = logging.getLogger("email_agent")
    logger.setLevel(level)
    
    # 清除现有的处理器
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(level)
    file_formatter = StructuredFormatter()
    file_handler.setFormatter(file_formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 获取日志记录器
logger = setup_logging()

def get_logger():
    """获取日志记录器"""
    return logger
