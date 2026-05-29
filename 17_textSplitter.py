from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. LOAD DOCUMENT
# Use TextLoader to read a local text file
print("--- Step 1: Loading Document ---")
loader = TextLoader("sample.txt")
docs = loader.load()

print(f"Loaded {len(docs)} document(s).")
print(f"Content length: {len(docs[0].page_content)} characters.")
print(f"Metadata: {docs[0].metadata}")

# 2. SPLIT DOCUMENT
# Use RecursiveCharacterTextSplitter to break the document into smaller chunks.
# This is crucial for LLMs with token limits.
print("\n--- Step 2: Splitting Document ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,      # Max characters per chunk
    chunk_overlap=20,    # Overlap between chunks to maintain context
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks.")

# 3. DISPLAY CHUNKS
print("\n--- Step 3: Displaying Chunks ---")
for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
    print(f"\nChunk {i+1}:")
    print("-" * 20)
    print(chunk.page_content)
    print("-" * 20)
    print(f"Metadata: {chunk.metadata}")

if len(chunks) > 5:
    print(f"\n... and {len(chunks) - 5} more chunks.")
