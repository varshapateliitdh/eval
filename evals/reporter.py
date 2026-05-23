"""
reporter.py — aggregates scored EvalResults and prints a readable report.

Output sections
---------------
1. Per-task table  — every task with its score and reason
2. Summary table   — pass rate by tag category
3. Overall score   — single headline number
"""

from collections import defaultdict
from evals.runner import EvalResult


def print_report(results: list[EvalResult]) -> None:
    """Print a full eval report to stdout."""

    _print_per_task_table(results)
    _print_category_summary(results)
    _print_overall_summary(results)


def _print_per_task_table(results: list[EvalResult]) -> None:
    col_id = 30
    col_type = 7
    col_score = 7
    col_reason = 60

    header = (
        f"{'Task ID':<{col_id}} "
        f"{'Type':<{col_type}} "
        f"{'Score':<{col_score}} "
        f"{'Reason':<{col_reason}}"
    )

    print("\n" + "=" * len(header))
    print("EVAL RESULTS — PER TASK")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        score_str = f"{r.score:.1f}" if r.score is not None else "N/A"
        reason = (r.score_reason or "Not scored")[:col_reason]
        print(
            f"{r.task.id:<{col_id}} "
            f"{r.task.eval_type:<{col_type}} "
            f"{score_str:<{col_score}} "
            f"{reason}"
        )

    print("=" * len(header))


def _print_category_summary(results: list[EvalResult]) -> None:
    """Group results by tag and print pass rate per category."""

    tag_scores: dict[str, list[float]] = defaultdict(list)

    for r in results:
        if r.score is not None:
            for tag in r.task.tags:
                tag_scores[tag].append(r.score)

    if not tag_scores:
        return

    print("\nSUMMARY BY CATEGORY")
    print("-" * 40)
    print(f"{'Category':<25} {'Pass Rate':>10}  {'Passed/Total':>14}")
    print("-" * 40)

    for tag in sorted(tag_scores):
        scores = tag_scores[tag]
        passed = sum(1 for s in scores if s == 1.0)
        total = len(scores)
        rate = (passed / total) * 100
        print(f"{tag:<25} {rate:>9.1f}%  {passed:>6}/{total:<6}")

    print("-" * 40)


def _print_overall_summary(results: list[EvalResult]) -> None:
    scored = [r for r in results if r.score is not None]
    if not scored:
        print("\nNo scored results.")
        return

    total = len(scored)
    passed = sum(1 for r in scored if r.score == 1.0)
    errors = sum(1 for r in results if r.error)
    rate = (passed / total) * 100

    print(
        f"\nOVERALL:  {passed}/{total} passed  ({rate:.1f}%)   errors={errors}")
    print()
