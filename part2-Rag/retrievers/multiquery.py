from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from embedding_config import get_embeddings
from llm_config import get_llm


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms."),
]

vectorstore = Chroma.from_documents(docs, get_embeddings())
retriever = vectorstore.as_retriever()
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=get_llm(),
)

query = "What is gradient descent?"
results = multi_query_retriever.invoke(query)
print("\nRetrieved Documents:\n")
for doc in results:
    print(doc.page_content)
