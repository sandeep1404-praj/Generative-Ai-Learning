"""Shared Groq chat-model configuration."""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Add your Groq API key to the .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )
