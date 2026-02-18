import os
import subprocess
import tempfile

from langchain_core.tools import tool

from models.analysis_result import AnalysisResult
from utils.code_refactorer import CodeRefactorer
from utils.github_searcher import GitHubSearcher
from utils.ruff_parser import RuffParser

_parser = RuffParser()
_STDIN_ARGS = ["--stdin-filename", "code.py", "-"]

_github_searcher = GitHubSearcher()
_code_refactorer = CodeRefactorer()


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
    return _code_refactorer.refactor_code(code=code, instructions=instructions)


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
    return _github_searcher.index_repositories(query, max_repos)


AGENT_TOOLS = [lint, format_code, refactor, run_tests, index_github_repositories]
