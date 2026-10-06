"""The held-out set has to actually be held out.

The claim this repository rests its conclusion on is that no system has seen the shifted
vocabulary. That claim is cheap to make in prose and easy to break by accident: one city added
to the gazetteer, one phrase reused between the two generators, and the experiment quietly stops
being the experiment while every number still prints.

So the separation is asserted here, token by token, rather than described.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from qlora_lab import rules
from qlora_lab.generate import generate
from qlora_lab.metrics import report
from qlora_lab.schema import STATES
from qlora_lab.shift import (
    HELD_OUT_CITIES,
    HELD_OUT_CLAUSES,
    HELD_OUT_OPENINGS,
    SHIFTED_SIZE,
    generate_shifted,
)

SHIFTED = generate_shifted()
MAIN = generate()

#: Pinned for the same reason the test split is: the conclusion is measured on these 200
#: records, and a change to the held-out vocabulary moves it.
FROZEN_SHIFTED_DIGEST = "84705a7a80d22ff14808a4d4087fadeafb6804722fe79f55f3ea6c002b251dd5"


class TestItIsActuallyHeldOut:
    def test_no_held_out_city_is_in_the_rule_gazetteer(self) -> None:
        """The rule baseline must return null for every one of them.

        If a city leaked into both, the rules would score on it and the comparison would be
        measuring something narrower than it claims.
        """
        known = {rules.fold(city) for city, _ in rules.GAZETTEER.values()}
        for city, _ in HELD_OUT_CITIES:
            assert rules.fold(city) not in known, f"{city} is in both"

    def test_the_held_out_states_are_real_federal_units(self) -> None:
        # Held out from the gazetteer, not invented: the schema still has to accept them.
        for _, uf in HELD_OUT_CITIES:
            assert uf in STATES

    def test_no_opening_phrase_is_shared_with_the_training_corpus(self) -> None:
        from qlora_lab.generate import OPENINGS

        trained = {phrase for phrases in OPENINGS.values() for phrase in phrases}
        held_out = {phrase for phrases in HELD_OUT_OPENINGS.values() for phrase in phrases}
        assert trained.isdisjoint(held_out)

    def test_the_perils_are_the_same_eight(self) -> None:
        from qlora_lab.generate import OPENINGS

        # Same task, different words. A different set of perils would be a different task.
        assert set(HELD_OUT_OPENINGS) == set(OPENINGS)

    def test_no_boolean_clause_is_shared(self) -> None:
        from qlora_lab.generate import (
            DRIVABLE_FALSE,
            DRIVABLE_TRUE,
            INJURIES_FALSE,
            INJURIES_TRUE,
            REPORT_FALSE,
            REPORT_TRUE,
            THIRD_PARTY_FALSE,
            THIRD_PARTY_TRUE,
        )

        trained = set(
            THIRD_PARTY_TRUE
            + THIRD_PARTY_FALSE
            + INJURIES_TRUE
            + INJURIES_FALSE
            + REPORT_TRUE
            + REPORT_FALSE
            + DRIVABLE_TRUE
            + DRIVABLE_FALSE
        )
        held_out = {p for pair in HELD_OUT_CLAUSES.values() for group in pair for p in group}
        assert trained.isdisjoint(held_out)

    def test_the_words_the_rules_look_for_are_absent(self) -> None:
        """The specific cues, not just the whole phrases.

        `assalto` is the commonest Brazilian word for a robbery and the rule set does not
        contain it; `rebocado` is what a tow is called and the rule set looks for `guincho`.
        If a held-out phrase happened to contain a rule cue, the shift would be smaller than it
        says, so the cue words themselves are checked.
        """
        cues = ("roubo", "roubaram", "furto", "furtaram", "guincho", "feridos", "machuc")
        corpus = " ".join(rules.fold(e.text) for e in SHIFTED)
        for cue in cues:
            assert rules.fold(cue) not in corpus, f"{cue!r} leaked into the held-out set"

    def test_no_message_is_shared_with_the_main_corpus(self) -> None:
        assert {e.text for e in SHIFTED}.isdisjoint({e.text for e in MAIN})


class TestTheLabelsAreStillExact:
    def test_the_size_is_what_the_report_says(self) -> None:
        assert len(SHIFTED) == SHIFTED_SIZE

    def test_generation_is_deterministic(self) -> None:
        assert [e.text for e in generate_shifted()] == [e.text for e in SHIFTED]

    def test_withheld_and_null_agree(self) -> None:
        for item in SHIFTED:
            assert set(item.withheld) == set(item.label.absent())

    def test_the_date_is_never_after_the_message_arrived(self) -> None:
        for item in SHIFTED:
            if item.label.occurred_on:
                assert item.label.occurred_on <= item.received_on

    def test_a_stolen_car_is_never_described_as_drivable(self) -> None:
        for item in SHIFTED:
            if item.label.peril in {"theft", "robbery"}:
                assert item.label.vehicle_drivable is None

    def test_the_split_is_named_so_nothing_mistakes_it_for_test(self) -> None:
        assert {e.split for e in SHIFTED} == {"shifted"}

    def test_the_identifiers_do_not_collide_with_the_main_corpus(self) -> None:
        assert {e.id for e in SHIFTED}.isdisjoint({e.id for e in MAIN})

    def test_it_has_not_moved(self) -> None:
        payload = "\n".join(
            json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) for e in SHIFTED
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert digest == FROZEN_SHIFTED_DIGEST, (
            "the held-out set changed. The generalisation result was measured on the old one."
        )


class TestTheRulesFallOver:
    """The result the shifted set exists to produce, defended as a floor.

    Not an exact number — that would have to be updated on every harmless change and would stop
    meaning anything — but the shape: the rules lose most of their accuracy, and they lose it by
    going silent rather than by inventing.
    """

    @pytest.fixture(scope="class")
    def scored(self):
        outputs = [rules.extract(e.text, e.received_on).to_target() for e in SHIFTED]
        return report("rules", [(e.id, e.label) for e in SHIFTED], outputs)

    def test_they_still_answer_with_valid_json(self, scored) -> None:
        # A rule set cannot produce malformed output; that is its one structural advantage.
        assert scored.json_validity.rate == 1.0

    def test_they_lose_most_of_their_field_accuracy(self, scored) -> None:
        assert scored.field_accuracy.rate < 0.60

    def test_almost_no_record_survives_intact(self, scored) -> None:
        assert scored.exact_record.rate < 0.05

    def test_they_fail_by_going_silent_rather_than_by_inventing(self, scored) -> None:
        """The failure mode that makes a rule set safe and useless at the same time.

        It has no way to decide that a sentence it does not recognise means something, so it
        returns null. That is the correct failure for a rule system to have, and it is why the
        comparison with a model has to report both numbers rather than one accuracy.
        """
        assert scored.asserted_absence.rate < 0.02
        assert scored.missed_value.rate > 0.50
