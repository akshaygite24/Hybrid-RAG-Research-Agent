from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from utils.llm import get_llm

def create_critic_agent() -> RunnableSerializable:
    llm = get_llm(temperature=0.3)
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a critical reviewer. Your job is to review and improve research answers.
         Review the answer for:
         - Missing important information
         - Weak or vague information
         - Poor structure or clarity
         - Factual gap
         
         Rules:
         - If the answer is good, return it as is with minor improvements
         - If the answer has issues, rewrite it to be clearer and more complete
         - Always return the final improved answer only
         - Do not add commentary like "here is the improved quality"
         - Keep the same topic and facts, only improved quality"""),

         ("human", """Original question: {question}
          answer to review: {answer}
          Please review and return an improved version of this answer.""")
    ])

    chain: RunnableSerializable = prompt | llm | parser
    return chain

def critique_answer(question: str, answer: str) -> str:
    try:
        chain: RunnableSerializable = create_critic_agent()

        result: str = chain.invoke({
            "question": question,
            "answer": answer
        })
        return result
    except Exception as e:
        print(f"Critic failed: {str(e)}")
        return answer
    

if __name__=="__main__":
    print("\n--- Test 1: Weak Answer ---")
    question = "What is climate change?"
    weak_answer = "Climate change is when climate changes."
    improved = critique_answer(question, weak_answer)
    print(f"\nOriginal Answer: {weak_answer}")
    print(f"\nImproved Answer: {improved}")

    print("\n--- Test 2: Good Answer ---")
    question = "What are renewable energy sources?"
    good_answer = """Renewable energy sources are energy sources that are naturally replenished. 
    The main types include solar energy, wind energy, hydroelectric power, geothermal energy, 
    and biomass. These sources produce little to no greenhouse gas emissions and are considered 
    key solutions to reducing dependence on fossil fuels and combating climate change."""
    improved = critique_answer(question, good_answer)
    print(f"\nOriginal Answer: {good_answer}")
    print(f"\nImproved Answer: {improved}")