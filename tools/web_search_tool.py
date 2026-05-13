from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from typing import Type

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the internet")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Use this to search the internet for current, recent, or latest information. "
        "Use when the user asks about recent events, latest trends, news, "
        "or anything that requires up-to-date information not found in documents."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as e:
            return f"Web search failed: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)
    
if __name__ == "__main__":
    tool = WebSearchTool()
    result = tool.run("Latest news in technology?")
    print(result)