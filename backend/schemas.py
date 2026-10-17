from pydantic import BaseModel, Field
from typing import Literal

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class ResearchRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] = Field(default_factory=list)
    use_planner: bool = False
    use_critic: bool = False
    