from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text_loader = TextLoader('sample.txt')
docs = text_loader.load()
text = docs[0].page_content

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    # separators=['\n\n', '\n', ' ', '']
)

spitted_text = splitter.split_text(text)

for i, chunk in enumerate(spitted_text):
    print(f'Chunk {i}: {chunk}')