from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

def get_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0):
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_kwargs={"parallel_tool_calls": False}
    )
    return llm