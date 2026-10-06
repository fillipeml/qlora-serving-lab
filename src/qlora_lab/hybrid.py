"""The system the measurements actually point at: the model reads, arithmetic is computed.

Neither of the two good systems wins everywhere, and they do not lose in the same places. Across
the validation split the fine-tuned adapter beats the rule set on every field that requires
reading Portuguese — the peril, the municipality, and all four three-state booleans — and loses
on exactly two:

    occurred_on            rules 99.5%   tuned 78.0%
    estimated_amount_brl   rules 97.5%   tuned 94.0%

Those two are not language. "Sexta passada" is a subtraction against a known date, and "três mil
e quinhentos" is a number written in words. A 0.5B transformer does them by pattern; eleven lines
of Python do them exactly, and will still do them exactly next year.

So the composition is: **take the model's record, and where deterministic code produced a value
for one of those two fields, use that instead.**

Two details carry the design.

**Where the rules are silent the model's answer stands.** A rule set has no way to decide that a
sentence it does not recognise means something, so on unfamiliar wording it returns null rather
than guessing. Overwriting the model's answer with that null would import the rule set's
brittleness into a system whose whole advantage is not having it. The `shifted` split is where
this is visible: the rules recover almost no dates there, and the hybrid keeps the model's.

**The two fields were chosen on validation, before the test split was looked at.** The same
pattern then appears on test, which is a confirmation rather than the reason. A composition
picked by reading per-field test scores would be fitted to the test set as surely as a threshold
would be, and would deserve none of the confidence the number suggests.

This costs no GPU. It is a function of two saved runs.
"""

from __future__ import annotations

from datetime import date

from .generate import Example
from .metrics import Report, report
from .schema import Notice, ParseError, parse

#: The fields where deterministic code is authoritative. Chosen on the validation split; see the
#: module docstring. Both are arithmetic rather than language, which is the reason and also the
#: reason not to expect the list to grow by itself.
ARITHMETIC_FIELDS: tuple[str, ...] = ("occurred_on", "estimated_amount_brl")


def compose(model: Notice, computed: Notice) -> Notice:
    """One record from both systems.

    Every field comes from the model except the arithmetic ones, and those come from the model
    too whenever the deterministic parser declined to answer.
    """
    overrides = {}
    for field in ARITHMETIC_FIELDS:
        value = getattr(computed, field)
        if value is not None:
            overrides[field] = value
    return Notice(
        peril=model.peril,
        occurred_on=overrides.get("occurred_on", model.occurred_on),
        city=model.city,
        state=model.state,
        third_party_involved=model.third_party_involved,
        injuries=model.injuries,
        police_report_filed=model.police_report_filed,
        vehicle_drivable=model.vehicle_drivable,
        estimated_amount_brl=overrides.get("estimated_amount_brl", model.estimated_amount_brl),
    )


def compose_outputs(examples: list[Example], model_outputs: list[str]) -> list[str]:
    """Compose a saved model run with the rule extractor, record by record.

    A model output that does not parse is passed through untouched. Repairing it from the rules
    would quietly turn a record the model failed to produce into one it half-produced, and the
    JSON validity column would then be measuring the rule set.
    """
    from . import rules

    if len(examples) != len(model_outputs):
        raise ValueError("one model output per record, in the same order")
    out: list[str] = []
    for example, raw in zip(examples, model_outputs, strict=True):
        try:
            model = parse(raw)
        except ParseError:
            out.append(raw)
            continue
        computed = rules.extract(example.text, example.received_on)
        out.append(compose(model, computed).to_target())
    return out


def score(name: str, examples: list[Example], model_outputs: list[str]) -> Report:
    composed = compose_outputs(examples, model_outputs)
    return report(name, [(e.id, e.label) for e in examples], composed)


def explain(example: Example, model: Notice, computed: Notice) -> list[str]:
    """Which of the two produced each arithmetic field, for one record."""
    lines = []
    for field in ARITHMETIC_FIELDS:
        theirs = getattr(computed, field)
        mine = getattr(model, field)
        truth = getattr(example.label, field)
        source = "computed" if theirs is not None else "model"
        chosen = theirs if theirs is not None else mine
        verdict = "correct" if chosen == truth else f"wrong, truth is {truth}"
        lines.append(f"  {field:<22} {source:<9} {_show(chosen):<14} {verdict}")
    return lines


def _show(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
