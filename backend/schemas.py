from pydantic import BaseModel, Field

class ResearchRequest(BaseModel):
    query: str
    chat_history: list[str] = Field(default_factory=list)
    use_planner: bool = False
    use_critic: bool = False
    