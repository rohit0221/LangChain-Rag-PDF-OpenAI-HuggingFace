import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_pinecone import PineconeVectorStore

load_dotenv()

from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

doc=read_doc('./documents/')
documents=chunk_data(docs=doc)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()


# Define Index Name
index_name = "rag-conversation-app"


# Checking Index
# if index_name not in pinecone.list_indexes():
#   # Create new Index
#   pinecone.create_index(name=index_name, metric="cosine", dimension=768)
#   docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
# else:
#   # Link to the existing index
#   docsearch = Pinecone.from_existing_index(index_name, embeddings)
  
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# Define the repo ID and connect to Mixtral model on Huggingface
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
  repo_id=repo_id, 
  model_kwargs={"temperature": 0.8, "top_k": 50}, 
  huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)

template = """
You are a fortune teller. These Human will ask you a questions about their life. 
Use following piece of context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer within 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 

"""

prompt = PromptTemplate(
  template=template, 
  input_variables=["context", "question"]
)

rag_chain = (
  {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
  | prompt 
  | llm
  | StrOutputParser() 
)

class ChatBot():
  load_dotenv()
  doc=read_doc('./documents/')
  documents=chunk_data(docs=doc)  
  
  # The rest of the code here

  rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )

def main():
    bot = ChatBot()
    #take input from user in python
    question = input("Ask a question: ")
    result =bot.rag_chain.invoke(question)
    print(result)

   
if __name__ == "__main__":
    main()