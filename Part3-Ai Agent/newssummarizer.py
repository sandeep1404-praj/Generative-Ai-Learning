import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings


# =========================
# Hugging Face Embeddings
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# Tavily Search Tool
# =========================

search_tool = TavilySearchResults(max_result=5)


# =========================
# Groq LLM
# =========================

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)


# =========================
# Prompt Template
# =========================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Summarize the following news into clear bullet points:

{news}
"""
)


# =========================
# Chain
# =========================

chain = prompt | llm | StrOutputParser()


news_result = search_tool.run("Latest AI news of 2026")

result = chain.invoke({
    "news": news_result
})

print(result)

print(search_tool.description)
print(search_tool.name)
print(search_tool.args)