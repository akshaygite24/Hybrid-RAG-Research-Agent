from utils.llm import llm

response = llm.invoke("What is AI?")

print(response.content)