"""
exact.py — scorer for deterministic calculation tasks.

Strategy
--------
We don't do string equality ("25%" == "25%") because the agent might say
"Your ROI is 25.00%" or "The return is 25%". Instead we:

  1. Extract all numbers from both the expected and agent output strings.
  2. Compare the primary number with a small tolerance (0.5%).

This handles formatting differences without needing a full LLM call.
"""

import re
from evals.runner import EvalResult


def _extract_number(text: str) -> float | None:
    """
    Pull the LAST numeric value out of a string.
    Handles negatives, decimals, and strips currency/percent symbols.

    We use the last number because agents typically restate the input values
    first and give the final answer at the end. For example:
        "With a price of $60 and EPS of $4, the P/E ratio is 15." → 15.0
        "You own 50 shares at $34.50, total value is $1,725.00."  → 1725.0
        "ROI: -20.00% (loss)"                                     → -20.0
    """
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if matches:
        return float(matches[-1])
    return None


def score_exact(result: EvalResult, tolerance: float = 0.5) -> EvalResult:
    """
    Score a single EvalResult using numeric extraction + tolerance comparison.

    Args:
        result   : EvalResult with agent_output filled in
        tolerance: how many percentage points of difference is still a pass
                   (default 0.5, meaning 25% and 25.3% both pass)

    Returns:
        The same EvalResult object with score and score_reason filled in.
        score = 1.0 (pass) or 0.0 (fail)
    """
    if result.error:
        result.score = 0.0
        result.score_reason = f"Agent error: {result.error}"
        return result

    expected_num = _extract_number(result.task.expected)
    actual_num = _extract_number(result.agent_output)

    if expected_num is None:
        # Fall back to case-insensitive substring match if no number in expected
        if result.task.expected.lower().strip() in result.agent_output.lower():
            result.score = 1.0
            result.score_reason = "Exact string match."
        else:
            result.score = 0.0
            result.score_reason = (
                f"Expected '{result.task.expected}' not found in output."
            )
        return result

    if actual_num is None:
        result.score = 0.0
        result.score_reason = f"No number found in agent output: '{result.agent_output[:100]}'"
        return result

    diff = abs(actual_num - expected_num)
    if diff <= tolerance:
        result.score = 1.0
        result.score_reason = (
            f"Pass. Expected {expected_num}, got {actual_num} (diff={diff:.3f})."
        )
    else:
        result.score = 0.0
        result.score_reason = (
            f"Fail. Expected {expected_num}, got {actual_num} (diff={diff:.3f} > tolerance {tolerance})."
        )

    return result


def score_all_exact(results: list[EvalResult]) -> list[EvalResult]:
    """Score all results that have eval_type == 'exact'."""
    for result in results:
        if result.task.eval_type == "exact":
            score_exact(result)
    return results
