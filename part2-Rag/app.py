"""Streamlit PDF RAG app using Groq and local Hugging Face embeddings."""
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from embedding_config import get_embeddings
from llm_config import get_llm

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "rag_documents"

st.set_page_config(page_title="RAG Book Assistant", page_icon="📚")
st.title("📚 RAG Book Assistant")
st.write("Upload a PDF, create its vector database, and ask questions using its content.")

if not os.getenv("GROQ_API_KEY"):
    st.warning("Add GROQ_API_KEY=your_key to your .env file, then restart the app.")

uploaded_file = st.file_uploader("Upload a PDF book", type=["pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Create / Rebuild Vector Database", type="primary"):
        temp_path = None
        try:
            with st.spinner("Reading PDF, creating embeddings, and saving to ChromaDB..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                documents = PyPDFLoader(temp_path).load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                chunks = splitter.split_documents(documents)
                if not chunks:
                    raise ValueError("No readable text was found in this PDF.")

                # Rebuild so documents from an earlier upload are not mixed in.
                if Path(PERSIST_DIRECTORY).exists():
                    shutil.rmtree(PERSIST_DIRECTORY)

                embeddings = get_embeddings()
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY,
                    collection_name=COLLECTION_NAME,
                )

            st.session_state["db_ready"] = True
            st.success(f"Vector database created successfully ({len(chunks)} chunks).")
        except Exception as exc:
            st.error(f"Could not create vector database: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

# Existing DB can also be used after restarting Streamlit.
db_exists = Path(PERSIST_DIRECTORY).exists() and st.session_state.get("db_ready", False)
if Path(PERSIST_DIRECTORY).exists() and not db_exists:
    # The database directory may be present from a previous app session.
    db_exists = True

if db_exists:
    st.divider()
    st.subheader("Ask Questions From the PDF")
    query = st.text_input("Enter your question")

    if query.strip():
        try:
            with st.spinner("Searching the document and generating an answer..."):
                embeddings = get_embeddings()
                vectorstore = Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=embeddings,
                    collection_name=COLLECTION_NAME,
                )
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful AI assistant. Answer using ONLY the
provided context. If the answer is not present in the context, say:
'I could not find the answer in the document.' Do not make up information."""),
                    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
                ])
                response = get_llm().invoke(
                    prompt.invoke({"context": context, "question": query})
                )

            st.subheader("AI Answer")
            st.write(response.content)
            with st.expander("Show retrieved document chunks"):
                for index, doc in enumerate(docs, start=1):
                    st.markdown(f"**Chunk {index}**")
                    st.write(doc.page_content)
                    st.caption(str(doc.metadata))
        except Exception as exc:
            st.error(f"Error while answering: {exc}")
else:
    st.info("Upload a PDF and click 'Create / Rebuild Vector Database' to get started.")
