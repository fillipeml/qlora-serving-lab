"""The composition: the model reads, deterministic code does the arithmetic.

Four properties, each of which the obvious implementation gets wrong.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from qlora_lab.generate import generate
from qlora_lab.hybrid import ARITHMETIC_FIELDS, compose, compose_outputs, score
from qlora_lab.metrics import report
from qlora_lab.schema import FIELDS, Notice

TEST = [e for e in generate() if e.split == "test"]


def _notice(**kwargs) -> Notice:
    base = {"peril": "collision"}
    base.update(kwargs)
    return Notice(**base)


class TestComposition:
    def test_the_computed_date_wins(self) -> None:
        model = _notice(occurred_on=date(2026, 3, 1))
        computed = _notice(occurred_on=date(2026, 3, 6))
        assert compose(model, computed).occurred_on == date(2026, 3, 6)

    def test_the_computed_amount_wins(self) -> None:
        model = _notice(estimated_amount_brl=Decimal("3000"))
        computed = _notice(estimated_amount_brl=Decimal("3500"))
        assert compose(model, computed).estimated_amount_brl == Decimal("3500")

    def test_where_the_rules_are_silent_the_model_stands(self) -> None:
        """The property that keeps the rule set's brittleness out of the hybrid.

        A rule set returns null on wording it does not recognise. Letting that null overwrite
        the model's answer would import exactly the failure the model is there to avoid, and it
        is what a naive "deterministic fields come from the deterministic system" would do.
        """
        model = _notice(occurred_on=date(2026, 3, 1), estimated_amount_brl=Decimal("3000"))
        silent = _notice()
        composed = compose(model, silent)
        assert composed.occurred_on == date(2026, 3, 1)
        assert composed.estimated_amount_brl == Decimal("3000")

    def test_nothing_but_the_two_arithmetic_fields_is_touched(self) -> None:
        model = _notice(
            peril="robbery",
            city="Recife",
            state="PE",
            third_party_involved=True,
            injuries=False,
            police_report_filed=True,
            vehicle_drivable=None,
        )
        # A rule record that disagrees about everything.
        computed = _notice(
            peril="theft",
            city="Goiânia",
            state="GO",
            third_party_involved=False,
            injuries=True,
            police_report_filed=False,
            vehicle_drivable=True,
        )
        composed = compose(model, computed)
        for field in FIELDS:
            if field in ARITHMETIC_FIELDS:
                continue
            assert getattr(composed, field) == getattr(model, field), field

    def test_the_declared_fields_are_the_ones_that_are_arithmetic(self) -> None:
        # Guards against the list quietly growing into language fields, where the measured
        # result is the other way round.
        assert set(ARITHMETIC_FIELDS) == {"occurred_on", "estimated_amount_brl"}


class TestComposingASavedRun:
    def test_an_unparseable_model_output_passes_through_untouched(self) -> None:
        """Otherwise JSON validity would start measuring the rule set.

        Repairing a failed generation from the rules turns a record the model did not produce
        into one it half-produced, and the validity column silently becomes a different
        statistic.
        """
        examples = TEST[:1]
        composed = compose_outputs(examples, ["I cannot help with that"])
        assert composed == ["I cannot help with that"]
        assert report("x", [(examples[0].id, examples[0].label)], composed).json_validity.rate == 0

    def test_it_refuses_a_run_of_the_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="one model output per record"):
            compose_outputs(TEST[:3], ["{}"])

    def test_composing_a_perfect_run_changes_nothing(self) -> None:
        # The ceiling case: if the model were already right, the rules cannot make it wrong,
        # because on this corpus they agree wherever they both answer.
        examples = TEST[:40]
        perfect = [e.label.to_target() for e in examples]
        scored = score("hybrid", examples, perfect)
        assert scored.exact_record.rate == 1.0

    def test_it_lifts_a_run_that_only_gets_the_dates_wrong(self) -> None:
        """The measured case, in miniature.

        Every record correct except the date, which is shifted by a day — which is what the
        adapter's failure actually looks like. The rule parser recovers it from the text.
        """
        from datetime import timedelta

        examples = [e for e in TEST[:60] if e.label.occurred_on is not None][:30]
        damaged = []
        for e in examples:
            wrong = Notice(
                peril=e.label.peril,
                occurred_on=e.label.occurred_on - timedelta(1),
                city=e.label.city,
                state=e.label.state,
                third_party_involved=e.label.third_party_involved,
                injuries=e.label.injuries,
                police_report_filed=e.label.police_report_filed,
                vehicle_drivable=e.label.vehicle_drivable,
                estimated_amount_brl=e.label.estimated_amount_brl,
            )
            damaged.append(wrong.to_target())

        before = report("model", [(e.id, e.label) for e in examples], damaged)
        after = score("hybrid", examples, damaged)
        assert before.exact_record.rate == 0.0
        assert after.exact_record.rate > 0.9
