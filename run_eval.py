"""
run_eval.py — entry point for the financial agent eval pipeline.

Usage
-----
    python run_eval.py                         # run all tasks
    python run_eval.py --tags calculation      # only tasks tagged 'calculation'
    python run_eval.py --tags live_data roi    # multiple tags (OR filter)
    python run_eval.py --task roi_basic        # single task by id

Pipeline flow
-------------
1. Load dataset  (evals/dataset.py)
2. Build agent   (agent/graph.py)
3. Run agent     (evals/runner.py)   ← real LLM + yfinance API calls
4. Score results (evals/scorers/)    ← exact match OR llm-judge
5. Print report  (evals/reporter.py)
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env if python-dotenv is available (optional convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Run financial agent evals")
    parser.add_argument(
        "--dataset",
        default="datasets/financial_qa.json",
        help="Path to the eval dataset JSON file",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Only run tasks with at least one of these tags",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Run a single task by its id",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge scoring (only run exact scorer)",
    )
    return parser.parse_args()


def check_env():
    """Fail fast with a clear message if required env vars are missing."""
    required = ["AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your Azure OpenAI details.")
        sys.exit(1)


def main():
    args = parse_args()
    check_env()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    from evals.dataset import load_dataset
    all_tasks = load_dataset(args.dataset)

    # Apply filters
    if args.task:
        tasks = [t for t in all_tasks if t.id == args.task]
        if not tasks:
            print(f"ERROR: No task with id '{args.task}' found.")
            sys.exit(1)
    elif args.tags:
        tasks = [t for t in all_tasks if any(
            tag in t.tags for tag in args.tags)]
        if not tasks:
            print(f"ERROR: No tasks match tags: {args.tags}")
            sys.exit(1)
    else:
        tasks = all_tasks

    print(f"\nLoaded {len(tasks)} task(s) from {args.dataset}")

    # ── 2. Build agent ───────────────────────────────────────────────────────
    print("Building agent...")
    from agent.graph import build_agent
    agent = build_agent()

    # ── 3. Run agent against dataset ─────────────────────────────────────────
    print(f"\nRunning {len(tasks)} task(s)...\n")
    from evals.runner import run_dataset
    results = run_dataset(agent, tasks, verbose=True)

    # ── 4. Score results ─────────────────────────────────────────────────────
    print("\nScoring results...")

    from evals.scorers.exact import score_all_exact
    results = score_all_exact(results)

    if not args.no_judge:
        judge_tasks = [r for r in results if r.task.eval_type == "judge"]
        if judge_tasks:
            from evals.scorers.llm_judge import build_judge_llm, score_all_judge
            print(f"  Running LLM judge on {len(judge_tasks)} task(s)...")
            judge_llm = build_judge_llm()
            results = score_all_judge(results, judge_llm)

    # ── 5. Print report ───────────────────────────────────────────────────────
    from evals.reporter import print_report
    print_report(results)


if __name__ == "__main__":
    main()
