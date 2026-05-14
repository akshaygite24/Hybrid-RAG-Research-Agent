from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from utils.llm import get_llm

class SubQuestions(BaseModel):
    questions: List[str] = Field(description="List of sub-questions derived from the main query")

def create_planner_agent():
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=SubQuestions)

    prompt = ChatPromptTemplate.from_messages([("system", """You are a research planner. Your job is to break down complex questions into smaller, focused sub-questions.
    Rules:
    - Break the query into maximum 3 specific sub-questions, never more
    - Each sub-questions should be self-sontained and answerable independently
    - If the query is already simple, return it as a single question
    - Always return valid JSON in this exact format:
    {{"questions": ["question 1","question 2","question 3"]}}
    - Do not add any explaination, only return the JSON"""),
    ("human", "Break down this query into sub-queries: {query}")
    ])

    chain = prompt | llm | parser
    return chain


def plan_query(query: str) -> List[str]:
    try:
        chain = create_planner_agent()
        result = chain.invoke({"query": query})

        if isinstance(result, dict):
            return result.get("questions", [query])
        
        return [query]
    except Exception as e:
        print(f"Planner failed: {str(e)}")
        return [query]
    

if __name__=="__main__":
    print("\n---- Test 1: Simple Query ---")
    question: List[str] = plan_query("What is climate change?")
    for i, q in enumerate(question):
        print(f"{i+1}. {q}")

    print("\n---- Test 2: Complex Query ---")
    question: List[str] = plan_query("Explain Climate change using my pdf and latest research trends?")
    for i, q in enumerate(question):
        print(f"{i+1}. {q}")

    print("\n---- Test 3: Multi-Topic Query ---")
    question: List[str] = plan_query("What is causing climate change and what are renewable energy solutions exist?")
    for i, q in enumerate(question):
        print(f"{i+1}. {q}")