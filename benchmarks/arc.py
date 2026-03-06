import re
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class ARCBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    @staticmethod
    def format_choices(problem: dict) -> str:
        """Format choices for inclusion in the prompt."""
        choices = problem["choices"]
        lines = []
        for label, text in zip(choices["label"], choices["text"]):
            lines.append(f"{label}. {text}")
        return "\n".join(lines)

    def extract_label(self, text: str) -> str:
        """Extract answer label (A-E) from model output.

        Handles short answers ("B", "D. a year"), CoT reasoning that echoes
        choice labels, and explicit answer markers ("Answer: D", "**C**").
        """
        text = text.strip()
        upper = text.upper()

        # 1. Single letter
        if len(upper) == 1 and upper in "ABCDE":
            return upper

        # 2. Starts with a valid label followed by non-letter
        if (
            upper
            and upper[0] in "ABCDE"
            and (len(upper) == 1 or not upper[1].isalpha())
        ):
            return upper[0]

        # 3. Explicit answer markers — take the last match
        #    Covers: "answer: D", "answer is B", "**C. text**", "**D**"
        marker = re.findall(
            r"(?:answer\s*(?:is|:)\s*\**\s*([A-E])\b|\*\*([A-E])[.\s)])", upper
        )
        if marker:
            last = marker[-1]
            return last[0] or last[1]

        # 4. Last standalone A-E letter (handles CoT that echoes choices)
        matches = re.findall(r"\b([A-E])\b", upper)
        if matches:
            return matches[-1]

        return text

    def calculate_score(
        self, expected_output: str, prediction: str
    ) -> Tuple[float, str]:
        extracted = self.extract_label(prediction)
        return (1.0 if extracted == expected_output else 0.0), extracted

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(
        self, problem: dict, graph: Callable
    ) -> Tuple[str, str, str, float, float]:
        question = problem["question"]
        choices_str = self.format_choices(problem)
        input_text = f"{question}\n\n{choices_str}"
        expected_output = problem["answerKey"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            score, extracted_output = self.calculate_score(expected_output, output)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
