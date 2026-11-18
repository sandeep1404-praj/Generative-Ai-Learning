
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


# Initialize Groq model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.9
)

print("Choose your AI mode")
print("Press 1 for Angry mode")
print("Press 2 for Funny mode")
print("Press 3 for Sad mode")

choice = input("Tell your response: ")

if choice == "1":
    mode = "You are an angry AI agent. You respond aggressively and impatiently."

elif choice == "2":
    mode = "You are a very funny AI agent. You respond with humor and jokes."

elif choice == "3":
    mode = "You are a very sad AI agent. You respond in a sad and emotional tone."

else:
    print("Invalid choice!")
    exit()


messages = [
    SystemMessage(content=mode)
]

print("----------------- Welcome! Type 0 to exit the application -----------------")

while True:

    prompt = input("You: ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)


print("\nConversation History:")
print(messages)
