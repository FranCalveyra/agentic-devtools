from agent.agent import agent
from src.config.config import config


def main():
    print(f"Initialized config: {config}")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print(result)


if __name__ == "__main__":
    main()
