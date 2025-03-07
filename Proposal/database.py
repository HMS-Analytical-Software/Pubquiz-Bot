import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import AzureOpenAIEmbeddings


from dotenv import load_dotenv
load_dotenv("../.env", override=True)


azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_KEY")

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=azure_key,
    openai_api_version="2024-06-01",
    azure_deployment=azure_deployment,
    azure_endpoint=azure_endpoint,
    chunk_size=16,
)

# load data
data = []
for file in os.listdir("../PubTexts/internal"):
    loader = TextLoader(f"../PubTexts/internal/{file}", encoding="utf-8")
    data.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=30000, chunk_overlap=1000, separators=[".", "\n"])
documents = splitter.split_documents(data)
db_internal = Chroma.from_documents(documents, embeddings, persist_directory="./chroma/internal")


data = []
for file in os.listdir("../PubTexts/reports"):
    loader = TextLoader(f"../PubTexts/reports/{file}", encoding="utf-8")
    data.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=30000, chunk_overlap=1000, separators=[".", "\n"])
documents = splitter.split_documents(data)
db_reports = Chroma.from_documents(documents, embeddings, persist_directory="./chroma/reports")


data = []
for file in os.listdir("../PubTexts/guides"):
    if file.endswith(".txt"):
        loader = TextLoader(f"../PubTexts/guides/{file}", encoding="utf-8")
        data.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=30000, chunk_overlap=1000, separators=[".", "\n"])
documents = splitter.split_documents(data)
db_guides = Chroma.from_documents(documents, embeddings, persist_directory="./chroma/guides")
