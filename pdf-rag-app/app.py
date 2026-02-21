"""
PDF RAG Application - Fully Local/Free Version
Uses HuggingFace embeddings and Groq's free LLM API
"""

# ----------------------------
# Load environment variables
# ----------------------------
from dotenv import load_dotenv
load_dotenv()

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
    page_icon="",
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

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)


def create_rag_chain(vector_store):
    groq_api_key = st.secrets["GROQ_API_KEY"]

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=groq_api_key,
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
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
st.title("PDF Q&A Assistant")

st.markdown(
    """
Upload a PDF document and ask questions about its content.  
This app uses **local embeddings** and **Groq's free LLM API**.
"""
)

with st.sidebar:
    st.header("Setup")

    st.markdown(
        """
### Groq API Key
This app reads your key from a `.env` file.

"""
    )

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing document..."):
            try:
                text = extract_text_from_pdf(uploaded_file)

                if not text.strip():
                    st.error(
                        "Could not extract text from PDF. "
                        "Please upload a text-based PDF."
                    )
                else:
                    vector_store = create_vector_store(text)
                    st.session_state.rag_chain = create_rag_chain(vector_store)
                    st.session_state.messages = []
                    st.success(" PDF processed successfully!")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    st.divider()

    st.markdown(
        """
###  Features
-  Local embeddings (no data sent for embeddings)
-  Free Groq API
-  Fast responses
-  Chat interface
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
                st.error(f"Error generating response: {e}")
else:
    st.info("Upload a PDF to begin.")

    with st.expander("How to use"):
        st.markdown(
            """
1. Create a `.env` file  
2. Add your `GROQ_API_KEY`
3. Run `streamlit run app.py`
4. Upload a PDF
5. Ask questions!
"""
)
