import streamlit as st
import wikipedia

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📚 Wikipedia RAG Chatbot (FAISS + TinyLlama)")

query = st.text_input("Enter topic (e.g. Artificial Intelligence):")

if query:

    with st.spinner("Loading Wikipedia..."):
        page = wikipedia.page(query)
        text = page.content

        doc = Document(page_content=text)

    # -------------------------------
    # SPLIT
    # -------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents([doc])

    # -------------------------------
    # EMBEDDINGS
    # -------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    # -------------------------------
    # FAISS INDEX
    # -------------------------------
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # -------------------------------
    # LLM
    # -------------------------------
    llm = pipeline(
        task="text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=300,
        temperature=0.7,
    )

    # -------------------------------
    # PROMPT
    # -------------------------------
    template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant.

Use ONLY the context below.
If answer is not in context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # -------------------------------
    # USER QUESTION
    # -------------------------------
    user_question = st.text_input("Ask a question about the topic:")

    if user_question:

        with st.spinner("Thinking..."):
            docs = retriever.invoke(user_question)

            context = "\n\n".join([d.page_content for d in docs])

            prompt = template.format(
                context=context,
                question=user_question
            )

            result = llm(prompt)

            answer = result[0]["generated_text"]

        st.subheader("🧠 Answer")
        st.write(answer)

        st.subheader("📄 Retrieved Context")
        for i, d in enumerate(docs):
            st.write(f"Chunk {i+1}")
            st.write(d.page_content[:300])
            st.write("---")