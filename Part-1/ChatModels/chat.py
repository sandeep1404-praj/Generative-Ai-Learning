
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

# Initialize Groq model
model = init_chat_model(
    "groq:openai/gpt-oss-20b",
    temperature=0.3
)

# Send prompt
response = model.invoke("Hello, explain Agentic AI.")

# Print response
print(response.content)
