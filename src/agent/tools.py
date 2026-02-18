import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request

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

# Lazy singleton — HuggingFace model is loaded only when the vector store is first used.
_vector_store = None


def _get_vector_store():
    """Return the shared CodeVectorStore singleton, creating it on first call."""
    global _vector_store
    if _vector_store is None:
        from rag.vector_store import CodeVectorStore

        _vector_store = CodeVectorStore()
    return _vector_store


def _rag_context(query: str) -> str:
    """Return RAG-retrieved code snippets, or an empty string if unavailable."""
    try:
        docs = _get_vector_store().as_retriever(k=3).invoke(query)
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


@tool
def index_github_repositories(query: str, max_repos: int = 3) -> str:
    """Searches GitHub for Python repositories matching a query and indexes their
    source code into the vector store so it can be used as context during refactoring.

    Args:
        query: Search terms describing the kind of repositories to find
            (e.g. "data validation", "async web framework", "CLI tools").
        max_repos: Number of top-starred repositories to index (default: 3, capped at 5)
    """
    from config import config
    from rag.code_splitter import CodeSplitter
    from rag.loader import RepositoryLoader

    token = config.environment.GITHUB_ACCESS_TOKEN
    if not token:
        return "Error: GITHUB_ACCESS_TOKEN is not set in .env — cannot search GitHub."

    max_repos = min(max_repos, 5)

    search_url = (
        "https://api.github.com/search/repositories"
        f"?q={urllib.parse.quote(query)}+language:python"
        f"&sort=stars&order=desc&per_page={max_repos}"
    )
    req = urllib.request.Request(
        search_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    repos = data.get("items", [])
    if not repos:
        return f"No repositories found for query: {query!r}"

    loader = RepositoryLoader()
    splitter = CodeSplitter()
    store = _get_vector_store()

    count = len(repos)
    summary_lines = [
        f"Indexing {count} repositor{'y' if count == 1 else 'ies'} for '{query}':"
    ]

    for repo in repos:
        full_name = repo["full_name"]
        owner = full_name.split("/", 1)[0]
        stars = repo.get("stargazers_count", 0)
        try:
            loader.load_repository(repository_name=full_name, creator=owner)
            docs = loader.get_repository_documents()
            chunks = splitter.split(docs)
            store.add_documents(chunks)
            summary_lines.append(
                f"  + {full_name} ({stars:,} stars) — {len(docs)} files,"
                + "{len(chunks)} chunks"
            )
        except Exception as exc:
            summary_lines.append(f"  - {full_name} — failed: {exc}")

    return "\n".join(summary_lines)


AGENT_TOOLS = [lint, format_code, refactor, run_tests, index_github_repositories]
