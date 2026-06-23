from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
# FIX 1: Correct import path for pulling from LangChain Hub
# from langchain_community import hub
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchResults

# 1. Initialize the Hugging Face Pipeline
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=500,
    temperature=0.7,
    return_full_text=False
)


user_input = input("Enter your query for the agent: ")
llm = HuggingFacePipeline(pipeline=pipe)

# FIX 2: Wrap the pipeline in ChatHuggingFace to support agent messaging structures
chat_model = ChatHuggingFace(llm=llm)

# 2. Define the Tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

search_tool = DuckDuckGoSearchResults()
tools = [add_numbers, multiply_numbers,search_tool]

# 3. Create the Agent
agent = create_agent(
    model=chat_model, # Use the chat wrapper here
    tools=tools,
    system_prompt="You are a helpful mathematical and friendly assistant.",
    checkpointer=InMemorySaver()
)

# 4. Invoke the agent by assigning a thread configuration for memory persistence
thread_config = {"configurable": {"thread_id": "session_01"}}

# Execute query requiring tool use
response = agent.invoke(
    {"messages": [{"role": "user", "content": user_input}]},
    thread_config
)

# Print the final AI response
print(response["messages"][-1].content)
