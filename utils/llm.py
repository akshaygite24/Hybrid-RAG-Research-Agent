from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

def get_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0):
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return llm