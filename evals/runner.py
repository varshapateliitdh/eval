"""
runner.py — runs the agent against every task in the dataset.

WHY keep this separate from scoring?
  - The runner is expensive (real LLM calls, real API calls).
  - The scorers are cheap (string matching or a second LLM call).
  - Separating them lets you re-score without re-running the agent.

Output
------
Returns a list of EvalResult objects — one per task — containing
the agent's raw output and tool call trace alongside the original task.
"""

import time
from dataclasses import dataclass, field
from evals.dataset import EvalTask


@dataclass
class EvalResult:
    """
    The result of running the agent on a single EvalTask.

    Fields
    ------
    task        : the original task definition
    agent_output: the agent's final text response
    tool_calls  : list of tool names the agent called (in order)
    error       : set if the agent raised an exception
    score       : filled in later by a scorer (None until scored)
    score_reason: explanation from the scorer (None until scored)
    """

    task: EvalTask
    agent_output: str = ""
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None
    score: float | None = None           # 0.0 = fail, 1.0 = pass
    score_reason: str | None = None


def run_dataset(agent, tasks: list[EvalTask], verbose: bool = True) -> list[EvalResult]:
    """
    Run the agent against every task and collect raw results.

    Args:
        agent  : the compiled LangGraph agent from build_agent()
        tasks  : list of EvalTask objects from load_dataset()
        verbose: if True, prints progress for each task

    Returns:
        list of EvalResult, one per task
    """
    from agent.graph import run_agent  # local import to avoid circular deps

    results = []

    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"[{i}/{len(tasks)}] Running: {task.id}")

        try:
            run = run_agent(agent, task.input)
            result = EvalResult(
                task=task,
                agent_output=run["output"],
                tool_calls=run["tool_calls"],
            )
        except Exception as e:
            result = EvalResult(
                task=task,
                error=str(e),
            )

        if verbose:
            status = "ERROR" if result.error else "ok"
            tools_used = ", ".join(
                result.tool_calls) if result.tool_calls else "none"
            print(f"         status={status}  tools={tools_used}")

        results.append(result)

        # Small delay between tasks to avoid Yahoo Finance rate limiting
        if i < len(tasks):
            time.sleep(1.5)

    return results
