from agent.agent import agent


def main():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print(result)


if __name__ == "__main__":
    main()
