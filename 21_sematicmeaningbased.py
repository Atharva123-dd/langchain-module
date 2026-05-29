from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# EMBEDDINGS (FIXED)
# -------------------------

model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

# -------------------------
# SEMANTIC CHUNKER
# -------------------------

chunker = SemanticChunker(
    model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5
)

# -------------------------
# TEXT
# -------------------------

text = """
Artificial Intelligence is transforming the modern world in many different ways.
AI systems are now used in healthcare, education, finance, cybersecurity, and transportation.

In healthcare, AI helps doctors detect diseases earlier and improve patient care.
Hospitals use intelligent systems to analyze medical scans and recommend treatments.

In education, AI-powered platforms create personalized learning experiences for students.
Students can learn at their own pace using adaptive learning systems.

Businesses use AI for customer support, recommendation systems, and market analysis.
Chatbots improve user experience and automate support.

AI also raises ethical concerns like privacy, bias, and job automation.
"""

# -------------------------
# SPLIT
# -------------------------

chunks = chunker.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk)