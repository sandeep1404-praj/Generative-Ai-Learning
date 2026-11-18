"""Command-line RAG assistant using Groq + Hugging Face + ChromaDB."""
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from embedding_config import get_embeddings
from llm_config import get_llm

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "rag_documents"


def main():
    if not Path(PERSIST_DIRECTORY).exists():
        print("Vector database not found. Run: python create_database.py")
        return

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
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Answer using ONLY the
provided context. If the answer is not present in the context, say:
'I could not find the answer in the document.' Do not make up information."""),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ])

    print("RAG system ready. Enter 0 to exit.")
    while True:
        query = input("You: ").strip()
        if query == "0":
            break
        if not query:
            continue

        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        response = llm.invoke(prompt.invoke({"context": context, "question": query}))
        print(f"\nAI: {response.content}\n")


if __name__ == "__main__":
    main()
