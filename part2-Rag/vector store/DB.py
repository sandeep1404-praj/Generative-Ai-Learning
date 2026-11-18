"""Small example showing how to save and query documents in ChromaDB."""
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from embedding_config import get_embeddings

PERSIST_DIRECTORY = "chroma_db_demo"


docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = get_embeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=PERSIST_DIRECTORY,
    collection_name="demo_documents",
)

print("Similarity search results:")
for document in vectorstore.similarity_search("what is used for data analysis?", k=2):
    print(document.page_content)
    print(document.metadata)

retriever = vectorstore.as_retriever()
print("\nRetriever results:")
for document in retriever.invoke("Explain deep learning"):
    print(document.page_content)
