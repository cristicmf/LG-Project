from langgraph.graph import Graph, END

def mock_llm(state):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = Graph()
graph.add_node("mock_llm", mock_llm)
graph.set_entry_point("mock_llm")
graph.set_finish_point("mock_llm")
graph = graph.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print("Result:", result)
