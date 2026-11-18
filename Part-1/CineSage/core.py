
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# Initialize Groq model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# Define Movie Schema
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)


# Create Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
Extract movie information from the paragraph.

Do not invent missing information.
Use null for unavailable optional fields.
Use empty lists when genre or cast information is unavailable.

{format_instructions}
"""),
    ("human", "{paragraph}")
])


# Get paragraph from user
para = input("Give your paragraph: ")

# Generate final prompt
final_prompt = prompt.invoke({
    "paragraph": para,
    "format_instructions": parser.get_format_instructions()
})

# Get response from Groq
response = model.invoke(final_prompt)

# Parse response into Pydantic object
movie_data = parser.parse(response.content)

print("\nExtracted Movie Information:")
print(movie_data.model_dump_json(indent=4))
