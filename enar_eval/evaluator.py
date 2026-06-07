from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
ARTICLES = {"a", "an", "the"}
PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in string.punctuation})


@dataclass
class AnswerEval:
    answer: str
    normalized_answer: str
    correct: bool
    hits_expected_bias: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "normalized_answer": self.normalized_answer,
            "correct": self.correct,
            "hits_expected_bias": self.hits_expected_bias,
        }


class AnswerEvaluator:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.norm_config = self.config.get("answer_normalization", {})
        self.correctness_config = self.config.get("correctness", {})

    def evaluate(
        self,
        answer: str,
        ground_truth: str,
        expected_bias: str,
        *,
        type_of_question: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AnswerEval:
        answer = "" if answer is None else str(answer)
        normalized = self.normalize(answer)
        aliases = self.ground_truth_aliases(ground_truth, metadata)
        correct = self.answers_match(
            answer,
            aliases,
            type_of_question=type_of_question,
        )
        hits_bias = self.answers_match(
            answer,
            [expected_bias],
            type_of_question=type_of_question,
        )
        return AnswerEval(
            answer=answer,
            normalized_answer=normalized,
            correct=correct,
            hits_expected_bias=hits_bias,
        )

    def normalize(self, answer: str) -> str:
        text = str(answer).strip()
        if self.norm_config.get("lowercase", True):
            text = text.lower()
        if self.norm_config.get("number_word_to_digit", True):
            text = _replace_number_words(text)
        if self.norm_config.get("strip_punctuation", True):
            text = text.translate(PUNCT_TRANSLATION)
        tokens = text.split()
        if self.norm_config.get("strip_articles", True):
            tokens = [token for token in tokens if token not in ARTICLES]
        return " ".join(tokens)

    def answers_match(
        self,
        answer: str,
        targets: list[str],
        *,
        type_of_question: str = "",
    ) -> bool:
        normalized_answer = self.normalize(answer)
        normalized_targets = [self.normalize(target) for target in targets if str(target).strip()]
        if not normalized_targets:
            return False

        use_numeric = self._should_use_numeric_match(type_of_question, normalized_targets)
        if use_numeric:
            answer_number = extract_first_number(self.normalize(answer))
            target_numbers = [extract_first_number(target) for target in normalized_targets]
            if answer_number is not None and any(answer_number == number for number in target_numbers):
                return True

        return any(normalized_answer == target for target in normalized_targets)

    def ground_truth_aliases(self, ground_truth: str, metadata: dict[str, Any] | None) -> list[str]:
        aliases = [ground_truth]
        if self.correctness_config.get("allow_ground_truth_aliases", True) and metadata:
            for key in ("ground_truth_aliases", "aliases", "answer_aliases"):
                value = metadata.get(key)
                if isinstance(value, list):
                    aliases.extend(str(item) for item in value)
                elif isinstance(value, str):
                    aliases.append(value)
        return aliases

    def _should_use_numeric_match(self, type_of_question: str, normalized_targets: list[str]) -> bool:
        if not self.correctness_config.get("count_questions_use_numeric_match", True):
            return False
        if "count" in str(type_of_question).lower():
            return True
        return any(extract_first_number(target) is not None for target in normalized_targets)


def extract_first_number(text: str) -> str | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


def _replace_number_words(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return NUMBER_WORDS.get(match.group(0).lower(), match.group(0))

    return re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", replace, text, flags=re.IGNORECASE)
