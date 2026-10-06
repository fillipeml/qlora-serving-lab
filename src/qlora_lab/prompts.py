"""What the model is told, and why the fine-tuned one is told almost nothing.

Three prompts, and the difference between them is a result rather than a detail. Median tokens
over 40 test records, counted with Qwen2.5's own tokeniser including its chat template:

    zero-shot      540    the instruction alone is 428 of them
    few-shot     1,175    the same instruction plus four worked examples
    fine-tuned      88    the message, the date, and where to start

The instruction is long because it has to be: it carries the schema, the eight enum values, the
date arithmetic and, at length, the rule that an unstated field is `null` and not `false`. A
fine-tuned model has all of that in its weights, and its prompt is **6.1x shorter than the
zero-shot one and 13.4x shorter than the few-shot one**. Those tokens come off the front of
every single request, which is why the serving benchmark reports prompt tokens beside latency:
on this hardware prefill dominates time to first token, and the cheapest way to make a model
faster is to stop explaining the job to it on every call.

The few-shot examples are drawn from the training split only, by a seeded sampler, and the same
four are used for every record. Sampling fresh examples per record would make the comparison
with the fine-tuned model meaningless — one system would have seen four times as much data as
the other, record by record.
"""

from __future__ import annotations

import random
from datetime import date

from .generate import Example
from .schema import PERILS

#: Where the model's answer begins. Shared by every prompt so one parser reads all three, and
#: so the fine-tuned model's target starts at exactly the same place.
ANSWER_PREFIX = "JSON:"

_PERIL_LIST = ", ".join(f'"{p}"' for p in PERILS)

SCHEMA_BLOCK = f"""{{
  "peril": one of [{_PERIL_LIST}],
  "occurred_on": "YYYY-MM-DD" or null,
  "city": string or null,
  "state": two-letter Brazilian state code or null,
  "third_party_involved": true, false or null,
  "injuries": true, false or null,
  "police_report_filed": true, false or null,
  "vehicle_drivable": true, false or null,
  "estimated_amount_brl": number or null
}}"""

INSTRUCTION = f"""You extract a structured record from a Brazilian insurance notice of loss.
The message is in Portuguese. The record is in English. Reply with the JSON object and nothing
else.

Schema:
{SCHEMA_BLOCK}

Rules:
- null means the message did not say. Never write false for a fact the message does not state.
  "Não houve feridos" is injuries=false. A message that never mentions injuries is
  injuries=null. The two are different facts and must not be merged.
- "roubo" (taken by force or threat) is "robbery". "furto" (taken without confrontation) is
  "theft". They are different perils.
- occurred_on is an absolute date. Resolve "ontem", "anteontem", "sexta passada" and
  "há N dias" against the date the message was received, which is given to you. "sexta passada"
  means the most recent Friday strictly before that date.
- city is the municipality named in the message. state is only filled when the message itself
  gives the two-letter code; do not infer it from the city.
- estimated_amount_brl is a number, without currency symbol or thousands separator. Amounts may
  be written in words ("três mil e quinhentos" is 3500).
- vehicle_drivable is false when the vehicle had to be towed or could not move."""


def _record_block(example: Example) -> str:
    return (
        f"Received on: {example.received_on.isoformat()}\nMessage: {example.text}\n{ANSWER_PREFIX}"
    )


def zero_shot(text: str, received_on: date) -> str:
    return (
        f"{INSTRUCTION}\n\nReceived on: {received_on.isoformat()}\nMessage: {text}\n{ANSWER_PREFIX}"
    )


def pick_shots(pool: list[Example], count: int = 4, seed: int = 7) -> list[Example]:
    """Four worked examples, fixed for the whole run.

    Stratified by peril rather than drawn uniformly: with eight perils and four slots, a uniform
    draw regularly gives two of the same, and which two would then be a confound in every
    comparison. Taking four distinct perils makes the choice arbitrary instead of lucky.
    """
    rng = random.Random(seed)
    by_peril: dict[str, list[Example]] = {}
    for item in pool:
        by_peril.setdefault(item.label.peril, []).append(item)
    perils = sorted(by_peril)
    rng.shuffle(perils)
    return [rng.choice(by_peril[p]) for p in perils[:count]]


def few_shot(text: str, received_on: date, shots: list[Example]) -> str:
    blocks = [INSTRUCTION, ""]
    for shot in shots:
        blocks.append(_record_block(shot) + " " + shot.label.to_target())
        blocks.append("")
    blocks.append(f"Received on: {received_on.isoformat()}\nMessage: {text}\n{ANSWER_PREFIX}")
    return "\n".join(blocks)


#: The whole prompt for a model that has been trained on the task. Everything the long
#: instruction says is in the weights; what is left is the two facts the model cannot know.
TUNED_INSTRUCTION = "Extract the claim record."


def tuned(text: str, received_on: date) -> str:
    return (
        f"{TUNED_INSTRUCTION}\n"
        f"Received on: {received_on.isoformat()}\n"
        f"Message: {text}\n"
        f"{ANSWER_PREFIX}"
    )


def training_pair(example: Example) -> tuple[str, str]:
    """Prompt and target for one training row.

    The target carries a leading space and a newline terminator: the space because the prompt
    ends at `JSON:` without one, and the newline because a model with no stop token learns to
    continue past the object and the generation has to be cut by length instead — which costs
    both latency and, when the continuation is another `{`, a parse failure.
    """
    return tuned(example.text, example.received_on), " " + example.label.to_target() + "\n"
