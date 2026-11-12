#load pdf
#split into chunks
#create the embeddings
#store into chromadb

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import  RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv

# load_dotenv()

# data = PyPDFLoader("document loaders/deeplearning.pdf")

# docs = data.load()

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks = splitter.split_documents(docs)

# embedding_model = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001"
# )

# vectorstore=Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     persist_directory="chroma_db"
# )


import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 1. Load PDF
data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

print("Total pages loaded:", len(docs))

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print("Total chunks generated:", len(chunks))

# 3. Create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 4. Initialize ChromaDB
vectorstore = Chroma(
    collection_name="rag_collection",
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

# Get already indexed documents
existing_data = vectorstore.get(include=["documents"])

existing_docs = set(existing_data["documents"])

# Find chunks that are not yet indexed
remaining_chunks = [
    chunk for chunk in chunks
    if chunk.page_content not in existing_docs
]

print("Already indexed:", len(existing_docs))
print("Remaining chunks:", len(remaining_chunks))

# 5. Add only remaining chunks in batches
batch_size = 20

for i in range(0, len(remaining_chunks), batch_size):

    batch = remaining_chunks[i:i + batch_size]

    vectorstore.add_documents(batch)

    print(
        f"Processed {min(i + batch_size, len(remaining_chunks))}"
        f"/{len(remaining_chunks)} remaining chunks"
    )

    if i + batch_size < len(remaining_chunks):
        print("Waiting 40 seconds...")
        time.sleep(40)

print("\nDatabase creation completed!")
print("Total indexed chunks:", vectorstore._collection.count())

# 5. Add chunks in batches
# batch_size = 50

# for i in range(0, len(chunks), batch_size):

#     batch = chunks[i:i + batch_size]

#     vectorstore.add_documents(batch)

#     print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

#     if i + batch_size < len(chunks):
#         print("Waiting 40 seconds...")
#         time.sleep(40)

# print("Database created successfully!")
