"""Scoring, split four ways, because "accuracy" hides the error that matters.

A single accuracy number treats these three failures as one:

    truth null, answered "false"      the file now asserts something nobody said
    truth "false", answered null      the file lost something somebody did say
    truth "false", answered "true"    the file is wrong

They are not one. The first is the only one that creates a fact out of nothing, and it is the
one a small model makes most and a single accuracy figure is least able to show — because the
same model that invents a value for an absent field can score well on the fields that are
present. So every field outcome is one of four states and every rate is reported per state.

`asserted_absence` is the headline. In an insurance file it is the difference between "no
injuries were reported" and "the claimant said there were no injuries", and nothing downstream
can tell them apart once it is written.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from decimal import Decimal
from typing import Any

from .schema import FIELDS, Notice, ParseError, parse
from .stats import Interval, wilson

CORRECT = "correct"
WRONG = "wrong"
ASSERTED_ABSENCE = "asserted_absence"
MISSED_VALUE = "missed_value"
STATUSES = (CORRECT, WRONG, ASSERTED_ABSENCE, MISSED_VALUE)


def _fold(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text.lower().strip())
        if unicodedata.category(c) != "Mn"
    )


def equivalent(field_name: str, truth: Any, predicted: Any) -> bool:
    """Whether two non-null values are the same answer.

    City is compared with accents folded and case ignored: a third of the corpus has had its
    accents stripped by the noise model, so a model that reads "Goiania" and writes "Goiania"
    has read it correctly, and scoring that as an error would measure the corpus rather than the
    model. Amounts are compared at two decimal places, dates exactly.
    """
    if field_name == "city":
        return _fold(str(truth)) == _fold(str(predicted))
    if field_name == "state":
        return str(truth).upper() == str(predicted).upper()
    if field_name == "estimated_amount_brl":
        return Decimal(str(truth)).quantize(Decimal("0.01")) == Decimal(str(predicted)).quantize(
            Decimal("0.01")
        )
    return truth == predicted


def classify(field_name: str, truth: Any, predicted: Any) -> str:
    if truth is None and predicted is None:
        return CORRECT
    if truth is None:
        return ASSERTED_ABSENCE
    if predicted is None:
        return MISSED_VALUE
    return CORRECT if equivalent(field_name, truth, predicted) else WRONG


@dataclass
class Scored:
    """One record's result."""

    id: str
    parsed: bool
    #: Why it did not parse, when it did not. Kept so a failure can be read rather than counted.
    parse_error: str | None = None
    outcomes: dict[str, str] = dataclass_field(default_factory=dict)

    @property
    def exact(self) -> bool:
        """Every field right. A record-level rate, which is what a claims handler experiences:
        one wrong field means the record is checked by hand, and it does not matter which."""
        return self.parsed and all(v == CORRECT for v in self.outcomes.values())

    def correct_fields(self) -> int:
        return sum(1 for v in self.outcomes.values() if v == CORRECT)


def score_one(example_id: str, truth: Notice, output: str) -> Scored:
    """Score one model output against one record.

    A record that does not parse is not scored field by field and does not quietly count as nine
    wrong answers: it counts as a record the system failed to produce, which is a different
    failure with a different fix. `field_accuracy` is therefore computed over parsed records
    only, and `json_validity` is reported beside it — reading either alone is a mistake the
    report is laid out to prevent.
    """
    try:
        predicted = parse(output)
    except ParseError as exc:
        return Scored(id=example_id, parsed=False, parse_error=str(exc))
    return Scored(
        id=example_id,
        parsed=True,
        outcomes={f: classify(f, getattr(truth, f), getattr(predicted, f)) for f in FIELDS},
    )


@dataclass(frozen=True)
class Report:
    """What one system did on one split."""

    system: str
    records: int
    json_validity: Interval
    exact_record: Interval
    field_accuracy: Interval
    asserted_absence: Interval
    missed_value: Interval
    per_field: dict[str, Interval]
    peril_accuracy: Interval
    #: Per-record exactness, in input order, for `stats.sign_test`.
    exact_by_record: list[bool]
    parse_errors: list[tuple[str, str]]

    def lines(self) -> list[str]:
        out = [
            f"{self.system}  ({self.records} records)",
            f"  JSON validity        {self.json_validity}",
            f"  exact record         {self.exact_record}",
            f"  field accuracy       {self.field_accuracy}   (parsed records only)",
            f"  asserted absence     {self.asserted_absence}   (of fields the text did not state)",
            f"  missed value         {self.missed_value}   (of fields the text did state)",
            f"  peril                {self.peril_accuracy}",
            "  per field:",
        ]
        for name, interval in self.per_field.items():
            out.append(f"    {name:24s} {interval}")
        if self.parse_errors:
            out.append(f"  first parse failures ({len(self.parse_errors)}):")
            for record_id, message in self.parse_errors[:3]:
                out.append(f"    {record_id}: {message}")
        return out


def report(system: str, truths: list[tuple[str, Notice]], outputs: list[str]) -> Report:
    if len(truths) != len(outputs):
        raise ValueError("one output per record, in the same order")
    scored = [score_one(i, t, o) for (i, t), o in zip(truths, outputs, strict=True)]
    parsed = [s for s in scored if s.parsed]

    absent_total = absent_asserted = present_total = present_missed = 0
    correct_fields = total_fields = 0
    per_field_correct: dict[str, int] = {f: 0 for f in FIELDS}
    peril_correct = 0
    for item, (_, truth) in zip(scored, truths, strict=True):
        if not item.parsed:
            continue
        for name, status in item.outcomes.items():
            total_fields += 1
            if status == CORRECT:
                correct_fields += 1
                per_field_correct[name] += 1
            if getattr(truth, name) is None:
                absent_total += 1
                if status == ASSERTED_ABSENCE:
                    absent_asserted += 1
            else:
                present_total += 1
                if status == MISSED_VALUE:
                    present_missed += 1
        if item.outcomes["peril"] == CORRECT:
            peril_correct += 1

    return Report(
        system=system,
        records=len(scored),
        json_validity=wilson(len(parsed), len(scored)),
        exact_record=wilson(sum(1 for s in scored if s.exact), len(scored)),
        field_accuracy=wilson(correct_fields, total_fields),
        asserted_absence=wilson(absent_asserted, absent_total),
        missed_value=wilson(present_missed, present_total),
        per_field={f: wilson(per_field_correct[f], len(parsed)) for f in FIELDS},
        peril_accuracy=wilson(peril_correct, len(parsed)),
        exact_by_record=[s.exact for s in scored],
        parse_errors=[(s.id, s.parse_error or "") for s in scored if not s.parsed],
    )


def confusion(truths: list[Notice], outputs: list[str], field_name: str = "peril") -> dict:
    """Which value was mistaken for which.

    Written for `peril`, where the interesting error is specific: Portuguese distinguishes
    `roubo` from `furto` and English does not, so a model trained mostly on English has every
    reason to collapse them. A single accuracy figure cannot show that; this can.
    """
    matrix: dict[tuple[Any, Any], int] = {}
    for truth, output in zip(truths, outputs, strict=True):
        try:
            predicted = parse(output)
        except ParseError:
            continue
        key = (getattr(truth, field_name), getattr(predicted, field_name))
        matrix[key] = matrix.get(key, 0) + 1
    return matrix
