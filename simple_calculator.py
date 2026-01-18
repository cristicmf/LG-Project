# 简单计算器示例（兼容 Python 3.8）

# 定义计算器函数
def add(a, b):
    """Adds `a` and `b`."""
    return a + b

def multiply(a, b):
    """Multiply `a` and `b`."""
    return a * b

def divide(a, b):
    """Divide `a` and `b`."""
    return a / b

# 定义处理函数
def calculate(message):
    """处理计算请求"""
    result = ""
    
    # 简单的命令解析
    if "add" in message.lower():
        # 提取数字
        import re
        numbers = re.findall(r'\d+', message)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            result = f"The result of adding {a} and {b} is {add(a, b)}."
        else:
            result = "Please provide two numbers to add."
    elif "multiply" in message.lower():
        import re
        numbers = re.findall(r'\d+', message)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            result = f"The result of multiplying {a} and {b} is {multiply(a, b)}."
        else:
            result = "Please provide two numbers to multiply."
    elif "divide" in message.lower():
        import re
        numbers = re.findall(r'\d+', message)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            if b != 0:
                result = f"The result of dividing {a} by {b} is {divide(a, b)}."
            else:
                result = "Cannot divide by zero."
        else:
            result = "Please provide two numbers to divide."
    else:
        result = "I can help you with addition, multiplication, and division. Please specify your calculation."
    
    return result

# 测试示例
print("=== 简单计算器示例 ===")

# 测试加法
result = calculate("Add 3 and 4.")
print("Input: Add 3 and 4.")
print("Output:", result)
print()

# 测试乘法
result = calculate("Multiply 5 and 6.")
print("Input: Multiply 5 and 6.")
print("Output:", result)
print()

# 测试除法
result = calculate("Divide 10 by 2.")
print("Input: Divide 10 by 2.")
print("Output:", result)
print()

# 测试无效输入
result = calculate("What's 123 plus 456?")
print("Input: What's 123 plus 456?")
print("Output:", result)
