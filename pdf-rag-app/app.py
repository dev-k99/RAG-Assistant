"""
PDF RAG Application - Fully Local/Free Version
Uses HuggingFace embeddings and Groq's free LLM API
"""

import os
import streamlit as st
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


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
    """Extract text from uploaded PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def create_vector_store(text):
    """Create FAISS vector store with local HuggingFace embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)

    # Using local HuggingFace embeddings - no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)


def create_rag_chain(vector_store, api_key):
    """Create RAG chain using Groq's free LLM API."""
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.3,
        groq_api_key=api_key,
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
st.markdown(
    """
    Upload a PDF document and ask questions about its content. 
    This app uses local embeddings and Groq's free API for responses.
    """
)

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown(
        """
        ### Get Your Free Groq API Key
        1. Visit [console.groq.com](https://console.groq.com)
        2. Sign up for a free account
        3. Generate an API key
        """
    )

    api_key = st.text_input("Groq API Key", type="password")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if api_key and uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing document... (This may take a moment on first run)"):
            try:
                text = extract_text_from_pdf(uploaded_file)
                
                if not text.strip():
                    st.error("Could not extract text from PDF. Please ensure it's a text-based PDF.")
                else:
                    vector_store = create_vector_store(text)
                    st.session_state.rag_chain = create_rag_chain(vector_store, api_key)
                    st.session_state.messages = []
                    st.success("✅ PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    st.divider()
    
    st.markdown(
        """
        ### 💡 Features
        - 🔒 Local embeddings (no data sent for embedding)
        - 🆓 Free Groq API (fast responses)
        - 💬 Chat interface
        - 📄 PDF text extraction
        """
    )


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
            try:
                answer = st.session_state.rag_chain.invoke(question)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.chat_message("assistant").write(answer)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
else:
    st.info("👆 Upload a PDF and enter your Groq API key to begin.")
    
    with st.expander("ℹ️ How to use"):
        st.markdown(
            """
            1. Get a free API key from [Groq](https://console.groq.com)
            2. Paste your API key in the sidebar
            3. Upload a PDF document
            4. Click "Process PDF"
            5. Start asking questions!
            """
        )