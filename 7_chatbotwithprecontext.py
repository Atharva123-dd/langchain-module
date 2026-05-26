from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=150,
    temperature=0.5,
        do_sample=False,
    return_full_text=False,
    clean_up_tokenization_spaces=False
)

llm = HuggingFacePipeline(pipeline=pipe)

chat_history = [
    SystemMessage(content="you are web developer assistant"),
]

print(" Simple AI Chat started (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Bye have a nice day!")
        break
    
    if user_input.strip()=="history":
        print("Chat History:")
        for msg in chat_history:
            print(f"{msg.__class__.__name__}: {msg.content}")
        continue
   

    full_input = f"""
You are a strict assistant.

RULES:
- Answer only the question
- Do NOT give examples
- Do NOT continue patterns
- Do NOT explain unless asked
history:
{chat_history}
Question: {user_input}
Answer:
"""

    response = llm.invoke(full_input)

    response = response.strip()
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    print(chat_history[-1].content)

    # chat_history.append({
    #     "role": "assistant",
    #     "content": response
    # })
    