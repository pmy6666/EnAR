from __future__ import annotations


class LlavaPromptBuilder:
    def build(self, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")
        return f"USER: <image>\n{question}\nASSISTANT:"
