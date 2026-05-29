from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text_loader = TextLoader('sample.txt')
docs = text_loader.load()
text = docs[0].page_content

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separator=''
)

spitted_text = splitter.split_text(text)

print(spitted_text)