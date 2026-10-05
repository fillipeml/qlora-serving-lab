"""The record a first notice of loss has to become, and what "the text did not say" means.

The source text is Brazilian Portuguese; the record is English, with English enum values. That
is not decoration: an insurer's downstream systems are in English, so the extraction is a
translation as well as a parse, and a model that only copies spans cannot do it. It is also the
cheapest honest test of whether a 0.6B model understands Portuguese or merely pattern-matches
it.

**Three states, not two.** Every optional field is `true`, `false`, or absent, and absent is not
`false`. "Não houve feridos" is `injuries: false` — somebody looked and said so. A message that
never mentions injuries is `injuries: null` — nobody looked. Writing `false` in the second case
puts a claim on file asserting something no human ever said, and the file gives no sign of it.
So the generator knows which fields it withheld, and the metrics count asserted absence
separately from an ordinary wrong answer.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

#: What happened. The message says it in Portuguese; the record says it in English.
PERILS: tuple[str, ...] = (
    "collision",
    # Portuguese separates `roubo` (taken by force or threat) from `furto` (taken without the
    # owner present). English collapses both into "theft", and an insurer does not: one is a
    # violent crime with a different reserve, a different fraud profile and a different police
    # process. A model that has only seen English has no reason to split them, which makes this
    # the single field that most directly tests whether it reads Portuguese or pattern-matches it.
    "robbery",
    "theft",
    "fire",
    "flood",
    "glass",
    "animal",
    "vandalism",
)

#: The 27 Brazilian federal units, for validating the `state` field.
STATES: frozenset[str] = frozenset(
    [
        "AC",
        "AL",
        "AP",
        "AM",
        "BA",
        "CE",
        "DF",
        "ES",
        "GO",
        "MA",
        "MT",
        "MS",
        "MG",
        "PA",
        "PB",
        "PR",
        "PE",
        "PI",
        "RJ",
        "RN",
        "RS",
        "RO",
        "RR",
        "SC",
        "SP",
        "SE",
        "TO",
    ]
)

#: The field order used everywhere: in the prompt, in the training target, in the report. Fixed
#: so a model never has to guess, and so two runs are comparable line by line.
FIELDS: tuple[str, ...] = (
    "peril",
    "occurred_on",
    "city",
    "state",
    "third_party_involved",
    "injuries",
    "police_report_filed",
    "vehicle_drivable",
    "estimated_amount_brl",
)

#: Fields whose absence is meaningful, i.e. everything except the one the message always states.
OPTIONAL: tuple[str, ...] = tuple(f for f in FIELDS if f != "peril")


@dataclass(frozen=True)
class Notice:
    """One extracted record. `None` means the text did not say."""

    peril: str
    occurred_on: date | None = None
    city: str | None = None
    state: str | None = None
    third_party_involved: bool | None = None
    injuries: bool | None = None
    police_report_filed: bool | None = None
    vehicle_drivable: bool | None = None
    estimated_amount_brl: Decimal | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field in FIELDS:
            value = getattr(self, field)
            if isinstance(value, date):
                out[field] = value.isoformat()
            elif isinstance(value, Decimal):
                # Two places, always. A bare float would print 3500.0 one run and
                # 3499.9999999999995 another, and the exact-match metric would move with it.
                out[field] = float(round(value, 2))
            else:
                out[field] = value
        return out

    def to_target(self) -> str:
        """The exact string the fine-tuned model is trained to emit.

        Compact separators and fixed key order: every token the model spends on whitespace is a
        token it does not spend on the answer, and at 6 GB of VRAM the sequence length is the
        budget.
        """
        return json.dumps(self.to_json(), ensure_ascii=False, separators=(",", ":"))

    def present(self) -> tuple[str, ...]:
        return tuple(f for f in FIELDS if getattr(self, f) is not None)

    def absent(self) -> tuple[str, ...]:
        return tuple(f for f in FIELDS if getattr(self, f) is None)


class ParseError(ValueError):
    """The model did not produce a record. Kept distinct from producing a wrong one."""


_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_TRUE = {"true", "sim", "yes", "1"}
_FALSE = {"false", "nao", "não", "no", "0"}


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    )


def _coerce_bool(value: Any, field: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        folded = _fold(value.strip())
        if folded in {"", "null", "none", "unknown", "nao informado", "n/a"}:
            return None
        if folded in _TRUE:
            return True
        if folded in _FALSE:
            return False
    raise ParseError(f"{field}: {value!r} is not a boolean")


def _coerce_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _fold(text) in {"null", "none", "unknown", "nao informado"}:
            return None
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            # A model that answers "12/03/2026" has understood the question and failed the
            # format; accepting it measures the understanding rather than the formatting.
            match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
            if match:
                day, month, year = (int(g) for g in match.groups())
                try:
                    return date(year, month, day)
                except ValueError as exc:
                    raise ParseError(f"occurred_on: {value!r} is not a date") from exc
    raise ParseError(f"occurred_on: {value!r} is not a date")


def _coerce_amount(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ParseError(f"estimated_amount_brl: {value!r} is not an amount")
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if _fold(text) in {"null", "none", "unknown", "nao informado"}:
            return None
        digits = re.sub(r"[^\d.,-]", "", text)
        # Brazilian notation: the comma is the decimal separator and the dot groups thousands.
        if "," in digits:
            digits = digits.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", digits):
            digits = digits.replace(".", "")
        try:
            return Decimal(digits)
        except InvalidOperation as exc:
            raise ParseError(f"estimated_amount_brl: {value!r} is not an amount") from exc
    raise ParseError(f"estimated_amount_brl: {value!r} is not an amount")


def parse(text: str) -> Notice:
    """Read a model's output into a record, or say why it is not one.

    Tolerant where tolerance measures the model and strict where it would hide a failure: a
    fenced code block, prose before the object and a trailing comma are formatting, so they are
    forgiven; an invented peril, an unknown field or a state that is not a federal unit are
    answers, so they are errors.
    """
    match = _OBJECT.search(text)
    if not match:
        raise ParseError("no JSON object in the output")
    body = match.group(0)
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", body)
        try:
            raw = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ParseError(f"the JSON object does not parse: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ParseError("the JSON value is not an object")

    unknown = set(raw) - set(FIELDS)
    if unknown:
        raise ParseError(f"fields that are not in the schema: {sorted(unknown)}")

    peril = raw.get("peril")
    if not isinstance(peril, str) or _fold(peril) not in PERILS:
        raise ParseError(f"peril: {peril!r} is not one of {list(PERILS)}")

    state = raw.get("state")
    if isinstance(state, str) and state.strip():
        state = state.strip().upper()
        if state not in STATES:
            raise ParseError(f"state: {state!r} is not a federal unit")
    else:
        state = None

    city = raw.get("city")
    city = city.strip() if isinstance(city, str) and city.strip() else None

    return Notice(
        peril=_fold(peril),
        occurred_on=_coerce_date(raw.get("occurred_on")),
        city=city,
        state=state,
        third_party_involved=_coerce_bool(raw.get("third_party_involved"), "third_party_involved"),
        injuries=_coerce_bool(raw.get("injuries"), "injuries"),
        police_report_filed=_coerce_bool(raw.get("police_report_filed"), "police_report_filed"),
        vehicle_drivable=_coerce_bool(raw.get("vehicle_drivable"), "vehicle_drivable"),
        estimated_amount_brl=_coerce_amount(raw.get("estimated_amount_brl")),
    )


def from_json(raw: dict[str, Any]) -> Notice:
    """Rebuild a record written by `to_json`, without the tolerance `parse` grants a model."""
    return Notice(
        peril=raw["peril"],
        occurred_on=date.fromisoformat(raw["occurred_on"]) if raw.get("occurred_on") else None,
        city=raw.get("city"),
        state=raw.get("state"),
        third_party_involved=raw.get("third_party_involved"),
        injuries=raw.get("injuries"),
        police_report_filed=raw.get("police_report_filed"),
        vehicle_drivable=raw.get("vehicle_drivable"),
        estimated_amount_brl=(
            Decimal(str(raw["estimated_amount_brl"]))
            if raw.get("estimated_amount_brl") is not None
            else None
        ),
    )


def as_dict(notice: Notice) -> dict[str, Any]:
    return asdict(notice)
