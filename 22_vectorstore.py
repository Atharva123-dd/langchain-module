from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document

doc1 = Document(
    page_content="A legendary opening batsman dominates world cricket with consistent centuries and fearless stroke play.",
    metadata={"year": 2011, "rating": 9.5, "role": "batsman", "format": "all-format"},
)

doc2 = Document(
    page_content="A fast bowler known for swinging the ball both ways terrorizes top batting lineups across conditions.",
    metadata={"year": 2015, "rating": 9.2, "role": "bowler", "format": "test cricket"},
)

doc3 = Document(
    page_content="A brilliant all-rounder contributes with both bat and ball, often turning matches single-handedly.",
    metadata={"year": 2018, "rating": 9.0, "role": "all-rounder", "format": "limited overs"},
)

doc4 = Document(
    page_content="A wicketkeeper with lightning-fast reflexes and sharp game awareness leads the team with great precision.",
    metadata={"year": 2020, "rating": 8.8, "role": "wicketkeeper", "format": "ODI"},
)

doc5 = Document(
    page_content="A young emerging cricketer rises through domestic leagues with explosive batting and fearless intent.",
    metadata={"year": 2024, "rating": 8.6, "role": "batsman", "format": "T20"},
)

docs=[doc1,doc2,doc3,doc4,doc5]

model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

from langchain_community.vectorstores import Chroma

vector_store = Chroma(
    embedding_function=model,
    persist_directory="chroma_db",
    collection_name="cricket_players"
)

vector_store.add_documents(docs)

vector_store.get(include=["metadatas", "documents"])

vector_store.similarity_search(
  query="batsman",
  k=2
)