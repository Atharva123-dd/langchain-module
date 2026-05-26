import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# ------------------------
# PAGE TITLE
# ------------------------
st.set_page_config(page_title="TinyLlama Chatbot")

st.title("🤖 TinyLlama LangChain Chatbot")

# ------------------------
# LOAD MODEL
# ------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200,
        temperature=0.7,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ------------------------
# PROMPT TEMPLATE
# ------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])
# ------------------------
# CREATE CHAIN
# ------------------------
chain = prompt | llm

# ------------------------
# USER INPUT
# ------------------------
user_input = st.text_input("Ask something:")

# ------------------------
# GENERATE RESPONSE
# ------------------------
if st.button("Generate"):
    if user_input.strip():

        with st.spinner("Thinking..."):
            response = chain.invoke({
                "input": user_input
            })

        st.success("Response Generated")
        st.write(response)

    else:
        st.warning("Please enter a question.")