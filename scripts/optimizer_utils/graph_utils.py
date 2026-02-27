import json
import os
import re
import time
import traceback
from typing import List, Set, Tuple

from scripts.prompts.optimize_prompt import (
    WORKFLOW_CUSTOM_USE,
    WORKFLOW_INPUT,
    WORKFLOW_OPTIMIZE_PROMPT,
    WORKFLOW_TEMPLATE,
)
from scripts.logs import logger


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph(self, round_number: int, workflows_path: str):
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.round_{round_number}.graph"

        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except ImportError as e:
            logger.error(f"Error loading graph for round {round_number}: {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(
            workflows_path, f"round_{round_number}", "prompt.py"
        )
        graph_file_path = os.path.join(
            workflows_path, f"round_{round_number}", "graph.py"
        )

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.error(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/workflows/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(
                id + 1, operator, path
            )
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(
        self, id: int, operator_name: str, file_path: str
    ) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            constructor = matched_data.get("constructor", f"{operator_name}(llm)")
            return f"{id}. {operator_name}: {desc}, with interface {interface}). Constructor: operator.{constructor} (requires self.llm)."

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_description: str,
        type: str,
        log_data: str,
    ) -> str:
        graph_input = WORKFLOW_INPUT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            operator_description=operator_description,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + WORKFLOW_CUSTOM_USE + graph_system

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.error(
                    f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})"
                )
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def _sanitize_prompt_content(self, content: str) -> str:
        """Convert optimizer's raw prompt output to valid Python assignments.

        The optimizer often generates prompts in a commented format like:
            # XXX_PROMPT = \"\"\"
            # prompt content here
            # \"\"\"
        which is invalid Python. This strips leading '# ' from each line
        to produce valid Python variable assignments.
        """
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.startswith("# "):
                cleaned_lines.append(line[2:])
            elif line == "#":
                cleaned_lines.append("")
            else:
                cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)

        # If result doesn't contain a variable assignment, wrap in a default
        if not re.search(r'\w+\s*=\s*"""', cleaned):
            cleaned = f'XXX_PROMPT = """\n{cleaned.strip()}\n"""'

        return cleaned

    @staticmethod
    def validate_prompt_references(
        graph_code: str, prompt_code: str
    ) -> Tuple[bool, Set[str], Set[str]]:
        """Validate that every prompt_custom.X reference in graph code has a definition in prompt code.

        Args:
            graph_code: The raw graph code (content of the <graph> tag, before WORKFLOW_TEMPLATE wrapping).
            prompt_code: The sanitized prompt code (after _sanitize_prompt_content).

        Returns:
            (valid, references, definitions) where valid is True if all references are satisfied.
        """
        references = set(
            re.findall(r"prompt_custom\.([A-Za-z_][A-Za-z0-9_]*)", graph_code)
        )
        # Filter out module dunder attributes
        references -= {
            "__name__",
            "__file__",
            "__doc__",
            "__spec__",
            "__loader__",
            "__package__",
            "__path__",
            "__cached__",
        }

        definitions = set(
            re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", prompt_code, re.MULTILINE)
        )

        missing = references - definitions
        return len(missing) == 0, references, definitions

    def write_graph_files(
        self, directory: str, response: dict, round_number: int, dataset: str
    ):
        graph = WORKFLOW_TEMPLATE.format(
            graph=response["graph"], round=round_number, dataset=dataset
        )

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        prompt_content = self._sanitize_prompt_content(response["prompt"])
        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(prompt_content)

        with open(
            os.path.join(directory, "__init__.py"), "w", encoding="utf-8"
        ) as file:
            file.write("")
