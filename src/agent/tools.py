import os
import subprocess
import tempfile

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from models.analysis_result import AnalysisResult
from utils.ruff_parser import RuffParser

_parser = RuffParser()
_STDIN_ARGS = ["--stdin-filename", "code.py", "-"]

# --- Refactor sub-chain ---

_refactor_llm = ChatOllama(model="llama3.1", temperature=0.2)

_REFACTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Python developer. Refactor the provided code to "
            "improve readability, follow PEP 8, use idiomatic Python, and apply "
            "type hints where appropriate.\n\n"
            "Return ONLY the refactored Python code — no explanation, no markdown "
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

_refactor_chain = _REFACTOR_PROMPT | _refactor_llm | StrOutputParser()

# Lazy singleton — HuggingFace model is loaded only when refactor is first called.
_vector_store = None


def _rag_context(query: str) -> str:
    """Return RAG-retrieved code snippets, or an empty string if unavailable."""
    global _vector_store
    if _vector_store is None:
        try:
            from rag.vector_store import CodeVectorStore

            _vector_store = CodeVectorStore()
        except Exception:
            return ""
    try:
        docs = _vector_store.as_retriever(k=3).invoke(query)
        return "\n\n---\n\n".join(d.page_content for d in docs)
    except Exception:
        return ""


# --- Tools ---


@tool
def lint(code: str) -> AnalysisResult:
    """Lints Python code with ruff and returns violations."""
    result = subprocess.run(
        ["ruff", "check", "--output-format", "json", *_STDIN_ARGS],
        input=code,
        capture_output=True,
        text=True,
    )
    violations = _parser.extract_violations(result.stdout)
    return AnalysisResult(code=code, violations=violations)


@tool
def format_code(code: str) -> str:
    """Formats Python code with ruff and returns the formatted version."""
    result = subprocess.run(
        ["ruff", "format", *_STDIN_ARGS],
        input=code,
        capture_output=True,
        text=True,
    )
    return result.stdout


@tool
def refactor(code: str, instructions: str = "") -> str:
    """Refactors Python code to improve readability, style, and idioms.

    Args:
        code: The Python code to refactor.
        instructions: Optional specific refactoring instructions
            (e.g. "use dataclasses", "split into smaller functions").
    """
    context = _rag_context(code[:500])
    instructions_section = (
        f"Specific instructions: {instructions}" if instructions else ""
    )
    context_section = (
        f"Similar code patterns for reference:\n```python\n{context}\n```"
        if context
        else ""
    )
    return _refactor_chain.invoke(
        {
            "code": code,
            "instructions_section": instructions_section,
            "context_section": context_section,
        }
    )


@tool
def run_tests(test_code: str) -> str:
    """Runs pytest on the provided Python test code and returns a pass/fail summary.

    Args:
        test_code: Self-contained, pytest-compatible Python test code.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_test.py", prefix="devtools_", delete=False
    ) as tmp:
        tmp.write(test_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["pytest", tmp_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).strip()
        status = "PASSED" if result.returncode == 0 else "FAILED"
        return f"Tests {status} (exit code {result.returncode})\n\n{output}"
    finally:
        os.unlink(tmp_path)


AGENT_TOOLS = [lint, format_code, refactor, run_tests]
