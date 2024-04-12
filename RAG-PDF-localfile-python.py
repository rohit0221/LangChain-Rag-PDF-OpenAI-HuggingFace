import os
from dotenv import load_dotenv
load_dotenv()

#write a function to load the document
def load_document(file_path):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

loaded_doc=load_document("documents\Budget_Speech.pdf")

# function to split the document into chunks
def  split_document(loaded_doc):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(loaded_doc)
    return chunks

chunks = split_document(loaded_doc)

#print(chunks)
# embed the document s\and create a vector store using the next few lines of code
def embed_document(chunks):
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    db = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return db

db = embed_document(chunks)







# write function to get a question from the user use that text to invoke the rag_chain
def get_question():
    question = input("Enter your question you need from your PDF document: ")
    return question

question = get_question()




# format the docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
#format_docs = format_docs(chunks)


# write function to create prompt for the rag chain
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

custom_rag_prompt = create_rag_prompt()



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


rag_chain=create_rag_chain(db, custom_rag_prompt)


print(rag_chain.invoke(question))











