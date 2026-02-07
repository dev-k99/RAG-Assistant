"""
PDF RAG Application - LangChain 1.x Compatible
"""

import os
import streamlit as st
from pypdf import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="wide",
)


# ----------------------------
# Helpers
# ----------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def create_vector_store(text, api_key):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)

   embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    return FAISS.from_texts(chunks, embeddings)


def create_rag_chain(vector_store, api_key):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
Use the provided context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = (
        {
            "context": retriever,
            "question": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# ----------------------------
# UI
# ----------------------------
st.title("📚 PDF Q&A Assistant")

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input("OpenAI API Key", type="password")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if api_key and uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing document..."):
            text = extract_text_from_pdf(uploaded_file)
            vector_store = create_vector_store(text, api_key)
            st.session_state.rag_chain = create_rag_chain(vector_store, api_key)
            st.session_state.messages = []
            st.success("PDF processed successfully!")


# ----------------------------
# Chat
# ----------------------------
if st.session_state.rag_chain:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Ask a question about the PDF"):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            answer = st.session_state.rag_chain.invoke(question)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.chat_message("assistant").write(answer)
else:
    st.info("Upload a PDF and enter your API key to begin.")
