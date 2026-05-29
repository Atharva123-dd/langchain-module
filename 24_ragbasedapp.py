from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_core.prompts import PromptTemplate

# -------------------------------
# LOAD PDF
# -------------------------------
loader = PyPDFLoader("one.pdf")
docs = loader.load()

# -------------------------------
# SPLIT INTO CHUNKS
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# -------------------------------
# EMBEDDINGS
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

# -------------------------------
# FAISS VECTOR STORE
# -------------------------------
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# LLM
# -------------------------------
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=300,
    temperature=0.7,
    return_full_text=False
)

# -------------------------------
# PROMPT
# -------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.

Use ONLY the context below:

{context}

Question:
{question}

Answer clearly and accurately:
"""
)

# -------------------------------
# RAG FUNCTION
# -------------------------------
def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    return llm(final_prompt.text)[0]["generated_text"]

# -------------------------------
# CHAT LOOP
# -------------------------------
while True:
    q = input("Ask: ")
    print(ask(q))