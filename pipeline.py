from langchain_core.messages import AIMessage, HumanMessage
from typing import List
from agents.planner_agent import plan_query
from agents.research_agent import create_research_agent, run_agent
from agents.critic_agent import critique_answer

def is_conversational(query: str) -> bool:
    query_lower = query.lower().strip()
    conversational = ["hi", "hello", "hey", "thanks", "bye", "how are you", "what's up", "thank you", "whats up", "good", "great"]
    # follow up - refers to previous conversation
    follow_up_keywords = ["summarize", "summary", "recap", "conclude", "conclude this", "summarize this", "what we did", "what have we", "we discussed", "tell me more", "elaborate", "explain more", "continue"]

    if query_lower in conversational:
        return True
    
    for keyword in follow_up_keywords:
        if query_lower.startswith(keyword):
            return True

    if len(query.split()) < 3:
        return True
    
    return False


def run_pipeline(query: str, chat_history: List = [], use_planner: bool = True, use_critic: bool = True) -> dict:
    print(f"\n{'='*50}")
    print(f"\nQuery: {query}")
    print(f"\n{'='*50}")

    if is_conversational(query):
        use_planner = False
        use_critic = False

    result = {
        "query": query,
        "sub_questions": [],
        "raw_answers": [],
        "final_answer": "",
        "used_planner": False,
        "used_critic": False
    }

    # Step 1 - Planner agent breaks querry into sub-question
    if use_planner:
        print(f"\n[Planner] Breaking query into sub-questions....")
        sub_questions: List[str] = plan_query(query)
        result["used_planner"] = True
    else:
        sub_questions: List[str] = [query]

    result["sub_questions"] = sub_questions
    print(f"[Planner] Sub-questions: {sub_questions}")

    # Step 2 - Research agent answers each sub question
    agent_executor = create_research_agent()
    raw_answers: List[str] = []

    for i, question in enumerate(sub_questions):
        print(f"\n[Research] Answering sub-question {i+1}: {question}")
        answer: str = run_agent(question, agent_executor, chat_history)
        raw_answers.append(answer)
        print(f"[Research] Answer {i+1}: {answer[:100]}")

    result["raw_answers"] = raw_answers

    # Step 3 - Combine all answers into one
    combined_answer: str = "\n\n".join(raw_answers)
    print(f"\n[Pipeline] Combined {len(raw_answers)} answers")

    # Step 4 - Critic improves the combined answer
    if use_critic:
        print(f"[Critic] Reviewing and Improving answer....")
        final_answer: str = critique_answer(query, combined_answer)
        result["used_critic"] = True
    else:
        final_answer: str = combined_answer

    result["final_answer"] = final_answer
    print(f"\n[Pipeline] Final answer ready")

    return result

if __name__=="__main__":
    chat_history = []

    print(f"\n---- Test 1: Full Pipeline ----")
    result = run_pipeline(
        query="What is climate change and what are the latest solutions in 2026?",
        chat_history=chat_history,
        use_planner=True,
        use_critic=True
    )
    print(f"\nFinal Answer:\n{result['final_answer']}")


    chat_history.append(HumanMessage(content="What is climate change and what are the latest solutions in 2026?"))
    chat_history.append(AIMessage(content=result['final_answer']))

    print(f"\n---- Test 2: Memory Test ----")
    result = run_pipeline(
        query="Summarize what we just discussed",
        chat_history=chat_history,
        use_planner=False,
        use_critic=False
    )
    print(f"\nFinal Answer:\n{result['final_answer']}")

