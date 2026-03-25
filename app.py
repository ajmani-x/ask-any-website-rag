import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

def load_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

def create_vector_db(splitted_docs):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    db = Chroma.from_documents(splitted_docs, embeddings, persist_directory="db/chroma_db")
    return db

def load_vector_db():
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    db = Chroma(persist_directory="db/chroma_db", embedding_function=embeddings)
    return db

def retrieve_query(query, db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    result = retriever.invoke(query)
    return result

def gen_result(context, query):
    model = ChatCohere(model="command-a-03-2025")
    prompt = f"""
You are a question answering system.

Use ONLY the provided context to answer the question.

If the context does not contain the answer, reply exactly:
"I don't know based on the provided context."

Context:
{context}

Question:
{query}

"""
    result = model.invoke(prompt)
    return result.content

st.title("Ask Any Website (RAG App)")
st.write("Enter a website URL, process it, and then ask questions about it.")

url = st.text_input("Enter Website URL")

if st.button("Process Website"):
    with st.spinner("Loading and indexing website..."):
        docs = load_docs(url)
        splitted_docs = split_docs(docs)
        create_vector_db(splitted_docs)
    st.success("Website processed successfully! You can now ask questions.")

st.divider()

query = st.text_input("Ask a question about the website")

if st.button("Get Answer"):
    if not url:
        st.warning("Please process a website first.")
    else:
        with st.spinner("Retrieving answer..."):
            db = load_vector_db()
            docs = retrieve_query(query, db)

        if not docs:
            st.write("No relevant information found in the website.")
        else:
            context = "\n\n".join([doc.page_content for doc in docs])
            result = gen_result(context, query)
            st.write(result)
