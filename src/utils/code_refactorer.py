from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from utils.vector_store_singleton import get_vector_store


class CodeRefactorer:
    def __init__(self) -> None:
        self._refactor_llm = ChatOllama(model="llama3.1", temperature=0.2)

        _REFACTOR_PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Python developer. Refactor the provided code to "
                    "improve readability, follow PEP 8, use idiomatic Python, and apply "  # noqa: E501
                    "type hints where appropriate.\n\n"
                    "Return ONLY the refactored Python code — no explanation, no markdown "  # noqa: E501
                    "fences, no commentary.",
                ),
                (
                    "human",
                    "Code to refactor:\n```python\n{code}\n```\n\n"
                    "{instructions_section}\n\n"
                    "{context_section}",
                ),
            ]
        )

        self._refactor_chain = _REFACTOR_PROMPT | self._refactor_llm | StrOutputParser()

    def _rag_context(self, query: str) -> str:
        """Return RAG-retrieved code snippets, or an empty string if unavailable."""
        try:
            docs = get_vector_store().as_retriever(k=3).invoke(query)
            return "\n\n---\n\n".join(d.page_content for d in docs)
        except Exception:
            return ""

    def refactor_code(self, code: str, instructions: str) -> str:
        context = self._rag_context(code[:500])

        instructions_section = (
            f"Specific instructions: {instructions}" if instructions else ""
        )
        context_section = (
            f"Similar code patterns for reference:\n```python\n{context}\n```"
            if context
            else ""
        )
        return self._refactor_chain.invoke(
            {
                "code": code,
                "instructions_section": instructions_section,
                "context_section": context_section,
            }
        )
