from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from embedding_config import get_embeddings


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms."),
]

vectorstore = Chroma.from_documents(docs, get_embeddings())

for search_type in ("similarity", "mmr"):
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": 3},
    )
    print(f"\n===== {search_type.upper()} Results =====\n")
    for doc in retriever.invoke("What is gradient descent?"):
        print(doc.page_content)
