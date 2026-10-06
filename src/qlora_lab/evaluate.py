"""Running one system over one split, and saying what it cost as well as what it scored.

Four systems, all answering the same 300 records:

    rules        regular expressions, no model, about a millisecond each
    zero-shot    the base model, told the rules at length
    few-shot     the base model, told the rules and shown four worked examples
    tuned        the same base model with a LoRA adapter, told almost nothing

Accuracy without latency is half a result. A system that is four points better and six times
slower is not better for an intake queue, and the table in the README is laid out so that
trade-off cannot be read past.

Outputs are saved verbatim next to the scores. Re-scoring a run must never require re-running
it: the metric changed twice while this repository was being written, and both times the fix
was to rescore recorded outputs rather than spend another hour of GPU.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .generate import Example
from .metrics import Report, report
from .schema import Notice


@dataclass
class Outcome:
    """One system's answers, scores and costs on one split."""

    system: str
    split: str
    outputs: list[str]
    report: Report
    #: Wall-clock milliseconds per record, end to end. For `rules` this is the whole cost; for a
    #: model it is prefill plus decode on this card at batch 1.
    median_ms: float
    median_ttft_ms: float
    decode_tokens_per_second: float
    prompt_tokens: int
    weight_gib: float
    notes: str = ""

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.system.replace('/', '_').replace(':', '-')}.{self.split}.json"
        path.write_text(
            json.dumps(
                {
                    "system": self.system,
                    "split": self.split,
                    "median_ms": self.median_ms,
                    "median_ttft_ms": self.median_ttft_ms,
                    "decode_tokens_per_second": self.decode_tokens_per_second,
                    "prompt_tokens": self.prompt_tokens,
                    "weight_gib": self.weight_gib,
                    "notes": self.notes,
                    "scores": {
                        "json_validity": str(self.report.json_validity),
                        "exact_record": str(self.report.exact_record),
                        "field_accuracy": str(self.report.field_accuracy),
                        "asserted_absence": str(self.report.asserted_absence),
                        "missed_value": str(self.report.missed_value),
                        "per_field": {k: str(v) for k, v in self.report.per_field.items()},
                    },
                    "exact_by_record": self.report.exact_by_record,
                    "outputs": self.outputs,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path


def _truths(examples: list[Example]) -> list[tuple[str, Notice]]:
    return [(e.id, e.label) for e in examples]


def run_rules(examples: list[Example], split: str) -> Outcome:
    from . import rules

    started = time.perf_counter()
    outputs = [rules.extract(e.text, e.received_on).to_target() for e in examples]
    elapsed_ms = (time.perf_counter() - started) * 1000
    return Outcome(
        system="rules",
        split=split,
        outputs=outputs,
        report=report("rules", _truths(examples), outputs),
        median_ms=elapsed_ms / max(len(examples), 1),
        median_ttft_ms=0.0,
        decode_tokens_per_second=0.0,
        prompt_tokens=0,
        weight_gib=0.0,
        notes="no model; written by the author of the generator, so read it as a ceiling",
    )


def run_model(
    examples: list[Example],
    split: str,
    model_id: str,
    style: str,
    precision: str = "nf4",
    adapter: str | None = None,
    shot_pool: list[Example] | None = None,
    batch_size: int = 1,
    # 110, because the target record is about 55 tokens. The cap only binds on a model that
    # does not know when to stop, and for that model every extra token is wasted wall clock.
    max_new_tokens: int = 110,
) -> Outcome:
    """Run one of the three prompting styles over a split.

    `style` is `zero-shot`, `few-shot` or `tuned`. The first two go through the model's chat
    template, because that is what an instruction-tuned model was trained to read; the third
    does not, because the adapter was trained on the raw completion format and a chat wrapper
    would put it in a context it has never seen.
    """
    from . import infer, prompts

    if style == "few-shot" and not shot_pool:
        raise ValueError("few-shot needs a pool of training examples to draw its shots from")

    loaded = infer.load(model_id, precision, adapter=adapter)
    try:
        if style == "zero-shot":
            texts = [prompts.zero_shot(e.text, e.received_on) for e in examples]
        elif style == "few-shot":
            shots = prompts.pick_shots(shot_pool or [])
            texts = [prompts.few_shot(e.text, e.received_on, shots) for e in examples]
        elif style == "tuned":
            texts = [prompts.tuned(e.text, e.received_on) for e in examples]
        else:
            raise ValueError(f"unknown style {style!r}")

        run = infer.generate(
            loaded,
            texts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            chat=style != "tuned",
        )
        totals = sorted(t.total_ms / max(batch_size, 1) for t in run.timings)
        name = f"{style}:{model_id.split('/')[-1]}:{precision}"
        return Outcome(
            system=name,
            split=split,
            outputs=run.outputs,
            report=report(name, _truths(examples), run.outputs),
            median_ms=totals[len(totals) // 2] if totals else 0.0,
            median_ttft_ms=run.median_ttft(),
            decode_tokens_per_second=run.median_decode_rate(),
            prompt_tokens=run.timings[0].prompt_tokens if run.timings else 0,
            weight_gib=loaded.weight_gib,
            notes=f"batch {batch_size}, greedy, adapter={adapter or 'none'}",
        )
    finally:
        infer.release(loaded)


def compare(left: Outcome, right: Outcome) -> str:
    """Whether the difference between two systems survives the pairing.

    Both ran on the same records in the same order, so the comparison is paired and the
    discordant records are the evidence. Two overlapping confidence intervals can still hide a
    difference that is consistent record by record, and comparing the point estimates alone
    would miss it in both directions.
    """
    from .stats import sign_test

    paired = sign_test(left.report.exact_by_record, right.report.exact_by_record)
    verdict = "significant at 0.05" if paired.p_value < 0.05 else "not significant at 0.05"
    return f"{left.system} -> {right.system}: {paired} ({verdict})"
