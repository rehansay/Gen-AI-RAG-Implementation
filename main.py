import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function=embedding_model,
    collection_name="rag_collection"
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

llm =ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)
print("Testing Gemini API...")

try:
    test_response = llm.invoke("Reply with just OK")
    print("Gemini test successful:", test_response.text)

except Exception as e:
    print("Gemini test failed:", repr(e))


#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)
    # docs = vectorstore.similarity_search(
    # query,
    # k=4
    # )
    print("\nRetrieved Documents:", len(docs))

    # for i, doc in enumerate(docs):
    #     print(f"\n--- Document {i+1} ---")
    #     print(doc.page_content[:500])

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.text}")

# try:
#     response = llm.invoke(final_prompt)
#     print(f"\nAI: {response.text}")

# except Exception as e:
#     print("\nError communicating with Gemini API.")
#     print("Error:", str(e))