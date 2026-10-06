"""One record, through every system, side by side.

Aggregates tell you a system is worse. They never tell you how. This reads the saved outputs —
no GPU, no model, nothing to re-run — and prints the message, the truth and each system's answer
with every field marked by which of the four things happened to it.

The marks are the whole point:

    =  correct
    x  wrong: the text said something and the system said something else
    +  asserted absence: the text said nothing and the system filled it in anyway
    -  missed value: the text said something and the system left it null

A model that looks four points behind on accuracy and makes all of its errors as `+` is not four
points behind. It is writing facts nobody stated into an insurance file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .generate import Example, read
from .metrics import ASSERTED_ABSENCE, CORRECT, MISSED_VALUE, WRONG, classify
from .schema import FIELDS, Notice, ParseError, parse

MARKS = {CORRECT: "=", WRONG: "x", ASSERTED_ABSENCE: "+", MISSED_VALUE: "-"}


def load_runs(directory: Path, split: str) -> list[dict[str, Any]]:
    """Every saved run for a split, in a stable order with the baseline first."""
    runs = []
    for path in sorted(directory.glob(f"*.{split}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split") == split and payload.get("outputs"):
            runs.append(payload)
    runs.sort(key=lambda r: (r["system"] != "rules", r["system"]))
    return runs


def _render(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def describe(example: Example, runs: list[dict[str, Any]], width: int = 22) -> list[str]:
    """The record, the truth and each system's answer, field by field."""
    lines = [
        f"{example.id}   received {example.received_on.isoformat()}",
        "",
        "  " + example.text,
        "",
    ]

    names = [r["system"] for r in runs]
    answers: list[Notice | None] = []
    for run in runs:
        try:
            index = run["outputs"]
            # The runs are saved in split order, so position is the key.
            answers.append(parse(index[example_index(run, example)]))
        except (ParseError, IndexError, KeyError):
            answers.append(None)

    header = f"  {'field':<22} {'truth':<20}" + "".join(f" {n[:width]:<{width}}" for n in names)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for field in FIELDS:
        truth = getattr(example.label, field)
        row = f"  {field:<22} {_render(truth):<20}"
        for answer in answers:
            if answer is None:
                row += f" {'(no record)':<{width}}"
                continue
            got = getattr(answer, field)
            mark = MARKS[classify(field, truth, got)]
            row += f" {mark} {_render(got)[: width - 2]:<{width - 2}}"
        lines.append(row)

    lines.append("")
    lines.append("  = correct   x wrong   + asserted absence   - missed value")
    return lines


#: Position of a record within a split, cached per split because every run shares it.
_POSITIONS: dict[str, dict[str, int]] = {}


def example_index(run: dict[str, Any], example: Example) -> int:
    split = run["split"]
    if split not in _POSITIONS:
        raise KeyError(f"positions for {split} were not loaded")
    return _POSITIONS[split][example.id]


def prepare(data: Path, split: str) -> list[Example]:
    examples = read(data, split)
    _POSITIONS[split] = {e.id: i for i, e in enumerate(examples)}
    return examples


def pick_disagreement(examples: list[Example], runs: list[dict[str, Any]]) -> Example | None:
    """A record the systems do not agree about, which is the only kind worth printing.

    Prefers one where the baseline is right and at least one model is wrong, because that is the
    comparison the report is actually making.
    """
    if not runs:
        return None
    by_system = {r["system"]: r.get("exact_by_record", []) for r in runs}
    baseline = by_system.get("rules")
    for index, example in enumerate(examples):
        flags = [v[index] for v in by_system.values() if index < len(v)]
        if len(set(flags)) < 2:
            continue
        if baseline is None or (index < len(baseline) and baseline[index]):
            return example
    return None
