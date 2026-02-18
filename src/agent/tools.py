import subprocess

from langchain_core.tools import tool

from models.analysis_result import AnalysisResult
from utils.ruff_parser import RuffParser

_parser = RuffParser()
_STDIN_ARGS = ["--stdin-filename", "code.py", "-"]


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


AGENT_TOOLS = [lint, format_code]
