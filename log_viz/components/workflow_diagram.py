"""Workflow operator flow diagram — Mermaid-based visualization of operator pipelines."""

import ast
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from data_loader import AFlowDataLoader


def _collect_names(node: ast.AST) -> set:
    """Collect all Name references in an AST subtree."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _get_assigned_vars(node: ast.AST) -> set:
    """Get variable names assigned by an Assign/AugAssign node."""
    if isinstance(node, ast.Assign):
        names = set()
        for target in node.targets:
            names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        return names
    if isinstance(node, ast.AugAssign):
        return {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    return set()


def _get_call_input_vars(call_node: ast.Call) -> set:
    """Get variable names referenced in call arguments/keywords."""
    names = set()
    for arg in call_node.args:
        names |= _collect_names(arg)
    for kw in call_node.keywords:
        names |= _collect_names(kw.value)
    names.discard("self")
    return names


def _collect_local_constants(body: list) -> Dict[str, int]:
    """Scan for simple `var = <int>` assignments to resolve variable loop counts."""
    constants: Dict[str, int] = {}
    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, int):
                    constants[target.id] = node.value.value
    return constants


def _get_loop_count(for_node: ast.For, local_vars: Optional[Dict[str, int]] = None):
    """Extract loop count from `for _ in range(N):` or `for _ in range(var):`."""
    if isinstance(for_node.iter, ast.Call):
        func = for_node.iter.func
        if isinstance(func, ast.Name) and func.id == "range":
            if for_node.iter.args:
                arg = for_node.iter.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    return arg.value
                if isinstance(arg, ast.Name) and local_vars:
                    val = local_vars.get(arg.id)
                    if isinstance(val, int):
                        return val
    return None


def _unwrap_await(node) -> Optional[ast.AST]:
    """Unwrap Await node to get the underlying call."""
    if isinstance(node, ast.Await):
        return node.value
    return None


def _get_source_lines(node: ast.AST, code_lines: List[str]) -> str:
    """Extract source code lines for an AST node."""
    start = getattr(node, "lineno", 0) - 1
    end = getattr(node, "end_lineno", start + 1)
    if 0 <= start < len(code_lines):
        return "\n".join(code_lines[start:end])
    return ""


def _extract_operator_class(operator_source: str, class_name: str) -> str:
    """Extract a single operator class definition from operator.py source."""
    try:
        tree = ast.parse(operator_source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            lines = operator_source.splitlines()
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end])
    return ""


def _extract_prompt_variable(prompt_source: str, var_name: str) -> str:
    """Extract a prompt variable assignment from op_prompt.py source."""
    try:
        tree = ast.parse(prompt_source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    lines = prompt_source.splitlines()
                    start = node.lineno - 1
                    end = node.end_lineno
                    return "\n".join(lines[start:end])
    return ""


def _classify_operator_call(call_node: ast.Call, _parent_node=None) -> Optional[Dict]:
    """Classify an operator call into a step dict."""
    func = call_node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    if func.value.id != "self":
        return None

    attr = func.attr
    op_map = {
        "custom": ("Custom", "Custom"),
        "answer_gen": ("AnswerGenerate", "Answer Generate"),
        "ensemble": ("ScEnsemble", "Sc Ensemble"),
    }

    if attr not in op_map:
        return None

    operator, label = op_map[attr]

    # Check if this is a review call (custom with REVIEW_PROMPT)
    detail = ""
    if attr == "custom":
        for kw in call_node.keywords:
            if kw.arg == "instruction":
                src = ast.dump(kw.value)
                if "REVIEW" in src:
                    operator = "Review"
                    label = "Self Review"
                    detail = "Expert review"

    return {
        "type": "operator",
        "operator": operator,
        "label": label,
        "detail": detail,
        "loop_count": None,
    }


def _parse_workflow_steps(code: str) -> List[Dict]:
    """Parse graph.py __call__ method to extract operator steps."""
    steps = []
    code_lines = code.splitlines()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return steps

    call_method = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "__call__":
                call_method = node
                break

    if call_method is None:
        return steps

    local_vars = _collect_local_constants(call_method.body)
    _extract_steps_from_body(call_method.body, steps, code_lines, local_vars)

    # Compute connectivity: does step[i] consume output of step[i-1]?
    for i in range(len(steps)):
        if i == 0:
            steps[i]["connected_to_prev"] = True
            continue
        prev_assigned = steps[i - 1].get("assigned_vars", set())
        cur_inputs = steps[i].get("input_vars", set())
        steps[i]["connected_to_prev"] = bool(prev_assigned & cur_inputs)

    return steps


def _extract_steps_from_body(
    body: list,
    steps: list,
    code_lines: List[str],
    local_vars: Optional[Dict[str, int]] = None,
) -> None:
    """Recursively extract operator call steps from AST body."""
    intermediate_deps: Dict[str, set] = {}

    def _resolve_vars(raw_inputs: set) -> set:
        resolved = set()
        for var in raw_inputs:
            if var in intermediate_deps:
                resolved |= intermediate_deps[var]
            else:
                resolved.add(var)
        return resolved

    for node in body:
        # For loop: look for operator calls inside
        if isinstance(node, ast.For):
            loop_count = _get_loop_count(node, local_vars)
            inner_steps: list = []
            _extract_steps_from_body(node.body, inner_steps, code_lines, local_vars)
            if inner_steps:
                # Track .append() calls to link list vars to inner step outputs
                inner_assigned = set()
                for s in inner_steps:
                    inner_assigned |= s.get("assigned_vars", set())
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if (
                            isinstance(func, ast.Attribute)
                            and func.attr == "append"
                            and isinstance(func.value, ast.Name)
                        ):
                            list_var = func.value.id
                            intermediate_deps[list_var] = (
                                intermediate_deps.get(list_var, set()) | inner_assigned
                            )

                for s in inner_steps:
                    # Only apply outer loop count if step isn't already in an inner loop
                    if s.get("loop_count") is None:
                        s["loop_count"] = loop_count
                        if loop_count and loop_count > 0:
                            s["label"] = f"{loop_count}x {s['label']}"
                    s["input_vars"] = _resolve_vars(s.get("input_vars", set()))
                steps.extend(inner_steps)
            continue

        # If statement: mark as conditional
        if isinstance(node, ast.If):
            source = _get_source_lines(node, code_lines)
            cond_inputs = _collect_names(node.test)
            steps.append(
                {
                    "type": "conditional",
                    "operator": "Conditional",
                    "label": "Review Check",
                    "detail": "if improved → re-generate",
                    "loop_count": None,
                    "source_line": source,
                    "assigned_vars": set(),
                    "input_vars": _resolve_vars(cond_inputs),
                }
            )
            # Also extract operator steps from the if body
            _extract_steps_from_body(node.body, steps, code_lines, local_vars)
            continue

        # Assignment with await: operator call
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            value = node.value if isinstance(node, ast.Assign) else node.value
            call_node = _unwrap_await(value)
            if call_node and isinstance(call_node, ast.Call):
                op_info = _classify_operator_call(call_node, node)
                if op_info:
                    assigned = _get_assigned_vars(node)
                    raw_inputs = _get_call_input_vars(call_node)
                    op_info["source_line"] = _get_source_lines(node, code_lines)
                    op_info["assigned_vars"] = assigned
                    op_info["input_vars"] = _resolve_vars(raw_inputs)
                    steps.append(op_info)
                    continue

            # Non-operator assignment — track as intermediate variable
            assigned = _get_assigned_vars(node)
            source_vars = _collect_names(value)
            source_vars -= assigned
            source_vars.discard("self")
            resolved = _resolve_vars(source_vars)
            for var in assigned:
                intermediate_deps[var] = resolved
            continue

        # Expression statement with await (no assignment)
        if isinstance(node, ast.Expr):
            call_node = _unwrap_await(node.value)
            if call_node and isinstance(call_node, ast.Call):
                op_info = _classify_operator_call(call_node, node)
                if op_info:
                    raw_inputs = _get_call_input_vars(call_node)
                    op_info["source_line"] = _get_source_lines(node, code_lines)
                    op_info["assigned_vars"] = set()
                    op_info["input_vars"] = _resolve_vars(raw_inputs)
                    steps.append(op_info)


def _generate_mermaid(steps: List[Dict]) -> str:
    """Generate a Mermaid flowchart from parsed workflow steps."""
    if not steps:
        return ""

    lines = ["graph TD"]

    # Define styles per operator type
    lines.append(
        "    classDef answerGen fill:#E3F2FD,stroke:#0091EA,color:#0D47A1,font-weight:bold"
    )
    lines.append(
        "    classDef scEnsemble fill:#E8F5E9,stroke:#00C853,color:#1B5E20,font-weight:bold"
    )
    lines.append(
        "    classDef review fill:#FFF3E0,stroke:#FF6F00,color:#E65100,font-weight:bold"
    )
    lines.append("    classDef conditional fill:#F5F5F5,stroke:#9E9E9E,color:#424242")
    lines.append(
        "    classDef custom fill:#FFF8E1,stroke:#FFD600,color:#F57F17,font-weight:bold"
    )

    class_map = {
        "AnswerGenerate": "answerGen",
        "ScEnsemble": "scEnsemble",
        "Review": "review",
        "Conditional": "conditional",
        "Custom": "custom",
    }

    # Build nodes and edges
    for i, step in enumerate(steps):
        node_id = f"S{i}"
        label = step["label"]
        cls = class_map.get(step["operator"], "")

        # Node shape: diamond for conditionals, rounded rectangle for operators
        if step["type"] == "conditional":
            node_def = f'{node_id}{{"{label}"}}'
        else:
            node_def = f'{node_id}["{label}"]'

        if i == 0:
            # First node — just define it
            lines.append(f"    {node_def}")
        else:
            # Edge from previous node
            prev_id = f"S{i - 1}"
            connected = step.get("connected_to_prev", True)
            arrow = "-->" if connected else "-.->"
            lines.append(f"    {prev_id} {arrow} {node_def}")

        if cls:
            lines.append(f"    class {node_id} {cls}")

    return "\n".join(lines)


def _render_mermaid(mermaid_code: str, height: int = 400) -> None:
    """Render a Mermaid diagram via HTML component."""
    html = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    </head>
    <body style="background: transparent; margin: 0; padding: 0;">
        <pre class="mermaid" style="display: flex; justify-content: center;">
{mermaid_code}
        </pre>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'neutral',
                flowchart: {{ curve: 'basis', padding: 10 }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html, height=height)


def display_workflow_diagram_single(
    loader: AFlowDataLoader,
    dataset: str,
    round_num: int,
) -> None:
    """Display a single round's workflow diagram with clickable steps."""
    code = loader.load_round_graph(dataset, round_num)
    if not code:
        st.info(f"No graph.py found for round {round_num}")
        return

    steps = _parse_workflow_steps(code)
    if not steps:
        st.info(f"No operator steps found for round {round_num}")
        return

    # Render Mermaid diagram
    mermaid_code = _generate_mermaid(steps)
    diagram_height = max(200, len(steps) * 70 + 60)
    _render_mermaid(mermaid_code, height=diagram_height)

    # Step selector — click a step to see its code
    step_labels = []
    for i, step in enumerate(steps):
        step_labels.append(f"Step {i + 1}: {step['label']}")

    selected = st.radio(
        "Inspect operator",
        step_labels,
        horizontal=True,
        key=f"step_select_r{round_num}",
    )
    selected_idx = step_labels.index(selected)
    step = steps[selected_idx]

    # Show call-site code from graph.py
    source_line = step.get("source_line", "")
    if source_line:
        st.markdown("**Call site** (from `graph.py`)")
        st.code(source_line, language="python")

    # Show operator class definition
    operator_source = loader.load_operator_source(dataset)
    if operator_source and step["operator"] not in ("Conditional",):
        class_code = _extract_operator_class(operator_source, step["operator"])
        if class_code:
            st.markdown(f"**Operator class** — `{step['operator']}`")
            st.code(class_code, language="python")

    # Show prompt template
    if step["operator"] in ("Custom", "Review"):
        round_prompt = loader.load_round_prompt(dataset, round_num)
        if round_prompt:
            st.markdown("**Custom prompt** (from `prompt.py`)")
            st.code(round_prompt, language="python")
    else:
        prompt_source = loader.load_operator_prompts(dataset)
        if prompt_source:
            prompt_map = {
                "AnswerGenerate": "ANSWER_GENERATION_PROMPT",
                "ScEnsemble": "SC_ENSEMBLE_PROMPT",
            }
            prompt_var = prompt_map.get(step["operator"])
            if prompt_var:
                prompt_text = _extract_prompt_variable(prompt_source, prompt_var)
                if prompt_text:
                    st.markdown(f"**Prompt template** — `{prompt_var}`")
                    st.code(prompt_text, language="python")
