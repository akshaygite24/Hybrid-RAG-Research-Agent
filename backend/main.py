from fastapi import FastAPI

from backend.schemas import ResearchRequest
from pipeline import run_pipeline

from langchain_core.messages import HumanMessage, AIMessage


app = FastAPI()


@app.get('/health')
def health():
    return {
        'status': 'OK',
        'message': 'Research Agent API is running'
    }

@app.post('/research')
def research(request: ResearchRequest):
    chat_history = [
        HumanMessage(content=message.content)
        if message.role=='user' 
        else AIMessage(content=message.content) 
        for message in request.chat_history
    ]

    result = run_pipeline(request.query, chat_history, request.use_planner, request.use_critic)

    return result