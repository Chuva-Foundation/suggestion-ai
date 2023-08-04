import os
import openai
import sys
import panel as pn  # GUI
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

sys.path.append('../..')

pn.extension()
load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

llm_name = "gpt-3.5-turbo"

# documents loader
loader = PyPDFLoader("./fake-db/cachupa.pdf")
docs = loader.load()

# print(docs[0])

# document split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
splits = text_splitter.split_documents(docs)

# print(len(splits))

# embeddings
embedding = OpenAIEmbeddings()

# vector_db
persist_directory = 'docs/chroma/'
#
# embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    persist_directory=persist_directory,
    embedding=embedding,
    documents=splits,
)
# print(vectordb._collection.count())

question = "Who is Stefan Vitoria?"
doc = vectordb.similarity_search(question,k=3)
print(len(docs))
vectordb.persist()
