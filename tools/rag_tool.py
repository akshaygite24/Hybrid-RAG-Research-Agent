from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from rag.retriever import retrieve_docs

class RAGInput(BaseModel):
    query: str = Field(description="The question to search in uploaded documents")

class RAGTool(BaseTool):
    name: str = "document_search"
    description: str = (
        "Use this to search and retrieve information from uploaded PDF documents. "
        "Use this when user mentions PDF, document, uploaded file, or any specific topic that might be in the document. "
        "Use when the user asks about content from their uploaded files or PDFs. "
    )
    args_schema: Type[BaseModel] = RAGInput

    def _run(self, query: str) -> str:
        try:
            docs = retrieve_docs(query)
            if not docs:
                return "No relevant information found in the uploaded documents."
            results = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                results.append(
                    f"[Chunk {i+1} | Source: {source} | Page: {page}]\n {doc.page_content.strip()}"
                )
            return "\n\n".join(results)
        except Exception as e:
            return f"Document search failed: {str(e)}"
        
    async def _arun(self, query: str) -> str:
        return self._run(query)
    
if __name__ =="__main__":
    tool = RAGTool()
    result = tool.run("What is climate change?")
    print(result)