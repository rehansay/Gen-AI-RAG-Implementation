from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
data = TextLoader("document loaders/notes.txt", encoding="utf-8")

splitter=CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1

)

docs = data.load()
chunks=splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()
    print()