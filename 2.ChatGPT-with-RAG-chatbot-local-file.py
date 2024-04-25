import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    pdf_reader= PdfReader(pdf_docs)
    text=""
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return  text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    return response

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using OpenAI💁")    

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        print(prompt)
        print(type(prompt))

        message=st.session_state.messages
        print(message)
        print(type(message))
        user_question = next((item['content'] for item in reversed(message) if item['role'] == 'user'), None)

        if user_question:
            answer=user_input(user_question)
            raw_text = get_pdf_text("documents\Budget_Speech.pdf")
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")


            #model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            st.session_state.messages.append({"role": "user", "content": user_question})
            print(answer)
            print(type(answer))


            msg = answer['output_text']
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg) 



if __name__ == "__main__":
    main()