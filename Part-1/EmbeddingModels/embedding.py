

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Hello this is Akarsh Vyas",
    "Hello your name is YouTube",
    "And you all are very beautiful"
]

# Generate embeddings
vectors = embeddings.embed_documents(texts)

print(vectors)
