"""
dataset.py — loads eval tasks from JSON into typed Python objects.

WHY: Working with raw dicts is error-prone. By loading into a Pydantic model
we get immediate validation (missing fields, wrong types) and autocomplete.
"""

import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel


class EvalTask(BaseModel):
    """
    Represents a single eval task.

    Fields
    ------
    id         : unique identifier used in results/reports
    input      : the user message sent to the agent
    eval_type  : "exact"  → compare agent output to expected string
                 "judge"  → ask an LLM to score based on criteria in expected
    expected   : for "exact": the correct answer string
                 for "judge": plain-English criteria the LLM judge checks
    tags       : arbitrary labels so you can slice results (e.g. ["roi", "calculation"])
    """

    id: str
    input: str
    eval_type: Literal["exact", "judge"]
    expected: str
    tags: list[str] = []


def load_dataset(path: str | Path) -> list[EvalTask]:
    """
    Load a JSON file of eval tasks and return a list of EvalTask objects.

    The JSON file must be a list of objects, each matching the EvalTask schema.
    Pydantic will raise a clear ValidationError if any task is malformed.
    """
    raw = json.loads(Path(path).read_text())
    return [EvalTask(**item) for item in raw]
