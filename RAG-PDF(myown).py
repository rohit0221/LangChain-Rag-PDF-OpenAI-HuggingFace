import os
import time
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
#openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
#openai_api_key=os.getenv('OPENAI_API_KEY')


def load_document(file_path):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages


def  split_document(loaded_doc):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
    )
    chunks = text_splitter.split_documents(loaded_doc)
    return chunks


def embed_document(chunks):
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    db = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return db

def handle_pdf():
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
         temp_file = "./temp.pdf"
         with open(temp_file, "wb") as file:
             file.write(uploaded_file.getvalue())
             file_name = uploaded_file.name
         with st.spinner(text='Loading the document...'):
             time.sleep(1) ## changed from 10 to 1
             st.success('Loading complete.')
         with st.spinner(text='Just a few minutes. I am processing it now...'):
             time.sleep(1) ## changed from 10 to 1
             pages=load_document(temp_file)
             chunks=split_document(pages)
             db = embed_document(chunks)             
             st.success('Congratulations! Your document is sane. Hang in there.')

         my_bar = st.progress(0)
         for percent_complete in range(1): ## changed from 100 to 1
             time.sleep(0.05)
             my_bar.progress(percent_complete + 1)
         st.success(' Processing completed. We are ready to chat with it.')
         st.balloons()    
         return db

        
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def get_question():
    with st.form('my_form'):
        text = st.text_area('Enter Question:', 'What do you want to ask the pdf?')
        submitted = st.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            question = text
            if question:
                st.write(f"You've asked: {question}")
            return question

def create_rag_prompt():
    from langchain_core.prompts import PromptTemplate

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    return custom_rag_prompt


def create_rag_chain(db, custom_rag_prompt):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = db.as_retriever()
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def generate_response(input_text,db, custom_rag_prompt):
    rag_chain=create_rag_chain(db, custom_rag_prompt)
    st.info(rag_chain.invoke(input_text))

st.title('🦜🔗 RAG Chain')
st.write("This is a simple web app to demonstrate the RAG Chain")
st.write("The RAG Chain is a tool that can be used to ask questions about a document")
st.title("Upload a PDF file")

db=handle_pdf()



if db is not None:
    question = get_question()
    if question is not None:
        #st.write("The question has been asked..preparing answer...")
        custom_rag_prompt = create_rag_prompt()
        if custom_rag_prompt:
                generate_response(question,db, custom_rag_prompt)
                #continue asking next questions
                question = get_question()

