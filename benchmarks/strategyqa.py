import re
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class StrategyQABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_yes_no(self, text: str) -> str:
        """Extract yes/no answer from model output."""
        text_lower = text.strip().lower()
        # Check if the answer starts with yes/no
        if text_lower.startswith("yes"):
            return "yes"
        if text_lower.startswith("no"):
            return "no"
        # Search for yes/no patterns in the text
        match = re.search(r"\b(yes|no)\b", text_lower)
        if match:
            return match.group(1)
        return text_lower

    def calculate_score(
        self, expected_output: str, prediction: str
    ) -> Tuple[float, str]:
        extracted = self.extract_yes_no(prediction)
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
        input_text = problem["question"]
        expected_output = problem["answer"]  # "yes" or "no"

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
