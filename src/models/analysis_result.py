from dataclasses import dataclass


@dataclass
class AnalysisResult:
    def __init__(self, code: str, violations: list[str]) -> None:
        self.code = code
        self.violations = violations

    def __str__(self) -> str:
        return (
            f"Resulting code for this analysis: {self.code}"
            + "\n"
            + f"The resulting SCA violations were: {self.violations}"
        )
