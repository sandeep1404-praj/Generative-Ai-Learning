"""Create/rebuild the Chroma vector database from the project PDF."""
from pathlib import Path
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from embedding_config import get_embeddings

PDF_PATH = Path("document loaders/deeplearning.pdf")
PERSIST_DIRECTORY = "chroma_db"


def create_database():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"Loading PDF: {PDF_PATH}")
    docs = PyPDFLoader(str(PDF_PATH)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text was extracted from the PDF.")

    # Remove any old database. Old OpenAI vectors cannot be queried using
    # the new Hugging Face embedding model.
    if Path(PERSIST_DIRECTORY).exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="rag_documents",
    )

    print(f"Vector database created with {len(chunks)} chunks.")


if __name__ == "__main__":
    create_database()
