# LangGraph 0.0.8 版本的计算器代理示例

# Step 1: 定义简单的计算器函数
def add(a, b):
    """Adds `a` and `b`."""
    return a + b

def multiply(a, b):
    """Multiply `a` and `b`."""
    return a * b

def divide(a, b):
    """Divide `a` and `b`."""
    return a / b

# Step 2: 定义处理函数
def calculate(state):
    """处理计算请求"""
    message = state["message"]
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
    
    return {"message": result}

# Step 3: 导入 LangGraph 组件
from langgraph.graph import Graph, END

# Step 4: 构建图
graph = Graph()
graph.add_node("calculate", calculate)
graph.set_entry_point("calculate")
graph.set_finish_point("calculate")

# Step 5: 编译图
graph = graph.compile()

# Step 6: 调用图
result = graph.invoke({"message": "Add 3 and 4."})
print("Result:", result)

# 测试其他计算
result = graph.invoke({"message": "Multiply 5 and 6."})
print("Result:", result)

result = graph.invoke({"message": "Divide 10 by 2."})
print("Result:", result)

