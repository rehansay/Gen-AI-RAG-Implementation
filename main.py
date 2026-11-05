
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import  RecursiveCharacterTextSplitter
load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

template=ChatPromptTemplate.from_messages(
    [("system","you are AI that summarizes the text"),
     ("human","{data}")]
)

chunks=splitter.split_documents(docs)

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


prompt=template.format_messages(data=docs)

try:
    result = model.invoke(prompt)
    print(result.content)

except Exception as e:
    print("Error:", repr(e))