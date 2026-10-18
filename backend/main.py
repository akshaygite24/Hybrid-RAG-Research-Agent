from fastapi import FastAPI, HTTPException

from backend.schemas import ResearchRequest, ResearchResponse
from pipeline import run_pipeline

from langchain_core.messages import HumanMessage, AIMessage


app = FastAPI()


@app.get('/health')
def health():
    return {
        'status': 'OK',
        'message': 'Research Agent API is running'
    }

@app.post('/research', response_model=ResearchResponse)
def research(request: ResearchRequest):
    chat_history = [
        HumanMessage(content=message.content)
        if message.role=='user' 
        else AIMessage(content=message.content) 
        for message in request.chat_history
    ]

    try:
        result = run_pipeline(
            request.query, 
            chat_history, 
            request.use_planner, 
            request.use_critic
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail='Research pipeline failed'
        )
    return result