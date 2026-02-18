from langchain_core.messages import AIMessage, ToolMessage

from agent.agent import agent

BANNER = """
╔══════════════════════════════════════════╗
║   Agentic DevTools  —  Python Assistant  ║
║                                          ║
║  Lint, format & improve your Python code ║
║  Type 'quit' or Ctrl-C to exit           ║
║  Enter a blank line to submit your input ║
╚══════════════════════════════════════════╝
"""

SEPARATOR = "─" * 44


def _read_user_input() -> str | None:
    """Read (potentially multi-line) input until a blank line is entered."""
    lines: list[str] = []
    try:
        while True:
            prompt = ">>> " if not lines else "... "
            line = input(prompt)
            if line == "" and lines:
                break
            lines.append(line)
    except EOFError:
        return None
    return "\n".join(lines).strip() or None


def _format_response(result: dict) -> str:
    """Extract a readable response from the agent's message list."""
    messages = result.get("messages", [])
    parts: list[str] = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            parts.append(f"  [tool: {msg.name}]  {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            parts.append(msg.content)

    return "\n\n".join(parts) if parts else str(result)


def main() -> None:
    print(BANNER)

    history: list[dict] = []

    while True:
        user_text = _read_user_input()
        if user_text is None or user_text.lower() in ("quit", "exit"):
            print("\nBye!")
            break

        history.append({"role": "user", "content": user_text})

        try:
            result = agent.invoke({"messages": history})
        except Exception as exc:
            print(f"\n⚠  Agent error: {exc}\n")
            continue

        print(f"\n{SEPARATOR}")
        print(_format_response(result))
        print(SEPARATOR)

        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                history.append({"role": "assistant", "content": msg.content})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
