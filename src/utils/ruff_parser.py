import json


class RuffParser:
    def extract_violations(self, lint_output: str) -> list[str]:
        violations = json.loads(lint_output or "[]")
        return [
            f"{v['location']['row']}:{v['location']['column']} "
            f"{v['code']}: {v['message']}"
            for v in violations
        ]
