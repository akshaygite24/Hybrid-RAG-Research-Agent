from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_llm
from tools.rag_tool import RAGTool
from tools.web_search_tool import WebSearchTool

rag_tool = RAGTool()
web_search_tool = WebSearchTool()

def create_research_agent():
    llm = get_llm()
    tools = [rag_tool, web_search_tool]

    prompt = ChatPromptTemplate.from_messages([("system", """You are a helpful research assistant. 
    You have access to two tools:
    1. document_search - search uploaded PDF documents
    2. web-search - search the internet for latest information
    Rules:
    - Use Document_search only when asked about uploaded PDFs or documemnts 
    - Use web_search only when asked about recent news or current events 
    - For conversational questions, greetings, summaries or follow-ups from chat history answer directly without calling any tool - If the answer is already in the chat history, use it directly"""), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}"),MessagesPlaceholder(variable_name="agent_scratchpad")])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10
    )

    return agent_executor

def run_agent(query: str, agent_executor: AgentExecutor,chat_history: list = []) -> str:
    try:
        response = agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })
        return response["output"]
    except Exception as e:
        return f"Agent failed: {str(e)}"


if __name__=="__main__":
    agent = create_research_agent()
    chat_history = []

    print("\n---- Test 1: Document Query ----")
    result = run_agent("What does the document say about climate change?", agent, chat_history)
    print(result)
    chat_history.append(HumanMessage(content="What does the document say about climate change?"))
    chat_history.append(AIMessage(content=result))

    print("\n---- Test 2: Web Search Query ----")
    result = run_agent("What are the latest technology news?", agent, chat_history)
    print(result)
    chat_history.append(HumanMessage(content="What are the latest technology news?"))
    chat_history.append(AIMessage(content=result))

    print("\n---- Test 3:  Memory Test ----")
    result = run_agent("Summarize what we just discussed", agent, chat_history)
    print(result)