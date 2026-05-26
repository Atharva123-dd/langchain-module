# pip install transformers torch accelerate langchain-huggingface langchain

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# Small model
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=200,
    temperature=0.7,
    return_full_text=False
)

# Convert to LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])

# Chain
chain = prompt | llm

# Run
response = chain.invoke({
    "input": "Explain LangChain in simple words"
})

print(response)

# from sqlalchemy import create_engine

# engine = create_engine("postgresql://postgres:admin@localhost:5432/langchain-test")

# conn = engine.connect()
# print("Connected!")