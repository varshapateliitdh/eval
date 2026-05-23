"""
llm_judge.py — scorer that uses an LLM to judge agent output quality.

WHY LLM-as-judge?
-----------------
For live data tasks (stock prices, P/E ratios), the correct answer changes
every day. We can't hardcode expected values. Instead, we write *criteria*
(e.g. "must include a plausible USD price for AAPL") and ask an LLM to
evaluate whether the agent's output meets those criteria.

The judge prompt is carefully structured to:
  - Return a clear PASS or FAIL verdict
  - Explain the reasoning
  - Not be fooled by confident-sounding but wrong answers

Judge model
-----------
We deliberately use a *separate* LLM call (not the agent itself) to judge.
Using the same model/session as the agent can lead to self-reinforcing bias.
"""

import os
import json
from langchain_openai import AzureChatOpenAI
from evals.runner import EvalResult


JUDGE_PROMPT_TEMPLATE = """\
You are an impartial evaluator assessing whether an AI assistant correctly answered a financial question.

## Task the assistant was given:
{task_input}

## Criteria for a correct answer:
{criteria}

## The assistant's actual response:
{agent_output}

## Tools the assistant called:
{tool_calls}

---

Evaluate whether the assistant's response meets ALL of the criteria above.
Be strict but fair. A response is a PASS only if it genuinely satisfies the criteria.

Respond ONLY with valid JSON in this exact format:
{{
  "verdict": "PASS" or "FAIL",
  "reason": "one or two sentences explaining your verdict"
}}
"""


def build_judge_llm() -> AzureChatOpenAI:
    """Create the LLM used as the judge (same Azure deployment as the agent)."""
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        temperature=0,  # always deterministic for judging
    )


def score_judge(result: EvalResult, judge_llm: AzureChatOpenAI) -> EvalResult:
    """
    Score a single EvalResult using LLM-as-judge.

    Sends the task input, criteria (from result.task.expected), agent output,
    and tool calls to the judge LLM and parses the PASS/FAIL verdict.

    Args:
        result    : EvalResult with agent_output and tool_calls filled in
        judge_llm : AzureChatOpenAI instance to use as the judge

    Returns:
        The same EvalResult with score (1.0 or 0.0) and score_reason filled in.
    """
    if result.error:
        result.score = 0.0
        result.score_reason = f"Agent error: {result.error}"
        return result

    tools_str = ", ".join(result.tool_calls) if result.tool_calls else "none"

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        task_input=result.task.input,
        criteria=result.task.expected,
        agent_output=result.agent_output,
        tool_calls=tools_str,
    )

    try:
        response = judge_llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if the LLM wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "").upper()
        reason = parsed.get("reason", "No reason provided.")

        result.score = 1.0 if verdict == "PASS" else 0.0
        result.score_reason = f"{verdict}: {reason}"

    except json.JSONDecodeError as e:
        result.score = 0.0
        result.score_reason = f"Judge returned malformed JSON: {e}. Raw: {raw[:200]}"
    except Exception as e:
        result.score = 0.0
        result.score_reason = f"Judge call failed: {e}"

    return result


def score_all_judge(results: list[EvalResult], judge_llm: AzureChatOpenAI) -> list[EvalResult]:
    """Score all results that have eval_type == 'judge'."""
    judge_tasks = [r for r in results if r.task.eval_type == "judge"]
    for i, result in enumerate(judge_tasks, 1):
        print(f"  Judging [{i}/{len(judge_tasks)}]: {result.task.id}")
        score_judge(result, judge_llm)
    return results
