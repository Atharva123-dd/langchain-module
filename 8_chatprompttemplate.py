from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder





chat_history = [
   
]
chat_template = ChatPromptTemplate(
    [
        ("system", "you are {domain} agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Explain in simple terms: {question}"),
    ]
)

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=100,
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)



print("🤖 Simple AI Chat started (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Bye 👋")
        break

    prompt = chat_template.invoke({"domain": "customer support", "question": user_input, "chat_history": chat_history})

    # IMPORTANT
    response = llm.invoke(prompt.to_string())

    response = response.strip()

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    print("Bot:", response)
