from langchain_huggingface import HuggingFacePipeline
import streamlit as st
from transformers import pipeline
from langchain_core.prompts import PromptTemplate,load_prompt

template =load_prompt("prompt_template.json")
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200,
        temperature=0.3,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()
st.header("LangChain with HuggingFace PROMPTTEMPLATE")
user_input = st.text_input("Enter your question:")

paper_input = st.selectbox(
    "Select a paper:",
    [
        "Paper 1: Deep Learning for NLP",
        "Paper 2: Advances in Computer Vision",
        "Paper 3: Reinforcement Learning Applications",
        "Paper 4: Generative AI and LLMs",
        "Paper 5: AI in Healthcare",
        "Paper 6: Quantum Computing Basics",
        "Paper 7: Cybersecurity with Machine Learning",
        "Paper 8: Blockchain Technology",
        "Paper 9: Data Science for Business",
        "Paper 10: Edge AI and IoT",
        "Paper 11: Robotics and Automation",
        "Paper 12: Explainable AI (XAI)",
        "Paper 13: Autonomous Vehicles",
        "Paper 14: Federated Learning",
        "Paper 15: Natural Language Understanding",
    ],
)

style_input = st.selectbox(
    "Select a response style:",
    [
        "Formal",
        "Informal",
        "Technical",
        "Layman's Terms",
        "Beginner Friendly",
        "Academic",
        "Research-Oriented",
        "Conversational",
        "Bullet Points",
        "Interview Style",
        "Storytelling",
        "Step-by-Step Explanation",
        "Professional",
        "Concise",
        "Detailed Analysis",
    ],
)

length_input = st.selectbox(
    "Select response length:",
    [
        "Very Short",
        "Short",
        "Medium",
        "Long",
        "Very Long",
        "Summary Only",
        "Detailed Report",
        "In-Depth Explanation",
        "One Paragraph",
        "Multi-Section Response",
    ],
)


if st.button("Generate Response"):
    chain = template | llm
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
        "question": user_input
    })
    st.write(result)
