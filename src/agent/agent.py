from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from agent.tools import get_weather

# We'll start with an Ollama model for developing purposes
llm = ChatOllama(model="llama3.1", temperature=0)
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
