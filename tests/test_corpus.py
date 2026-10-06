"""The corpus, the schema and the rule baseline — everything that runs without a GPU.

These are the tests CI can execute. The GPU half of the repository cannot run on a hosted
runner, and pretending otherwise with mocks would assert that the mocks behave, so it is not
tested here; what is tested is every claim that does not need a card, including the ones the
README makes about the data.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from qlora_lab import prompts, rules
from qlora_lab.generate import (
    PROTECTED,
    SPLITS,
    Example,
    add_noise,
    generate,
    spell_amount,
)
from qlora_lab.metrics import (
    ASSERTED_ABSENCE,
    CORRECT,
    MISSED_VALUE,
    WRONG,
    classify,
    report,
)
from qlora_lab.schema import FIELDS, Notice, ParseError, parse
from qlora_lab.stats import sign_test, wilson

CORPUS = generate()


@pytest.fixture(scope="session")
def corpus() -> list[Example]:
    return CORPUS


class TestTheCorpusIsFrozen:
    """The test split must be the same 300 records in every run of this repository's history.

    Every number in the README is measured on it. A change to the generator — a new template, a
    different probability, one more city — reshuffles the draw and moves every result, and the
    report would be comparing systems evaluated on different data without saying so. The digest
    below is the promise; this test is what keeps it from being only prose.

    If this fails after a deliberate change to the generator, the digest is updated in the same
    commit as the re-measured README, never on its own.
    """

    def test_the_split_sizes_are_what_the_report_says(self, corpus) -> None:
        counts = {split: sum(1 for e in corpus if e.split == split) for split, _ in SPLITS}
        assert counts == {"train": 1000, "validation": 200, "test": 300}

    def test_generation_is_deterministic(self) -> None:
        assert [e.text for e in generate()] == [e.text for e in CORPUS]

    def test_a_different_seed_gives_a_different_corpus(self) -> None:
        assert [e.text for e in generate(seed=1)] != [e.text for e in CORPUS]

    def test_no_message_appears_in_two_splits(self, corpus) -> None:
        # Leakage here would make the fine-tuned model's test score partly a memory test.
        by_split: dict[str, set[str]] = {}
        for item in corpus:
            by_split.setdefault(item.split, set()).add(item.text)
        assert by_split["train"].isdisjoint(by_split["test"])
        assert by_split["train"].isdisjoint(by_split["validation"])
        assert by_split["validation"].isdisjoint(by_split["test"])

    def test_the_test_split_has_not_moved(self, corpus) -> None:
        test = [e for e in corpus if e.split == "test"]
        payload = "\n".join(
            json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) for e in test
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert digest == FROZEN_TEST_DIGEST, (
            "the test split changed. Every number in the README was measured on the old one. "
            "Re-measure and update both in the same commit, or revert the generator."
        )


class TestTheLabelsAreTrue:
    def test_every_withheld_field_is_null_and_every_null_field_is_withheld(self, corpus) -> None:
        # The two have to agree or `asserted_absence` is measuring the wrong denominator.
        for item in corpus:
            assert set(item.withheld) == set(item.label.absent())

    def test_the_peril_is_never_withheld(self, corpus) -> None:
        assert all(e.label.peril for e in corpus)

    def test_state_is_never_filled_without_a_city(self, corpus) -> None:
        assert not any(e.label.state and not e.label.city for e in corpus)

    def test_a_stolen_car_is_never_described_as_drivable(self, corpus) -> None:
        # The generator refuses to write a record a person could not have written. Without this
        # the corpus teaches a model something false.
        for item in corpus:
            if item.label.peril in {"theft", "robbery"}:
                assert item.label.vehicle_drivable is None

    def test_only_a_collision_carries_another_party(self, corpus) -> None:
        for item in corpus:
            if item.label.third_party_involved is True:
                assert item.label.peril in {"collision", "animal"}

    def test_the_date_is_never_after_the_message_arrived(self, corpus) -> None:
        for item in corpus:
            if item.label.occurred_on:
                assert item.label.occurred_on <= item.received_on

    def test_every_stated_amount_is_recoverable_from_the_text(self, corpus) -> None:
        """The label has to be readable from the message, or it is not a label.

        The whole corpus design rests on the record being exact by construction. This is the
        one invariant that checks it rather than assuming it, and it caught a real defect: the
        loose amount style wrote `value // 1000` unconditionally, so 5,900 was rendered as
        "uns 5 mil" while the label said 5900. 37 of 772 records carried a message contradicting
        their own label, and every system was marked wrong on all of them for reading correctly.
        """
        from qlora_lab.generate import spell_amount

        for item in corpus:
            value = item.label.estimated_amount_brl
            if value is None:
                continue
            amount = int(value)
            folded = _fold(item.text).lower()
            forms = [
                f"{amount:,}".replace(",", "."),  # R$ 3.500,00
                _fold(spell_amount(amount)).lower(),  # três mil e quinhentos
                str(amount),  # 3500 reais
            ]
            if amount % 1000 == 0:
                forms.append(f"{amount // 1000} mil")
            assert any(form in folded for form in forms), (
                f"{item.id} is labelled {amount} and its message does not say so: {item.text}"
            )

    def test_the_amount_matches_the_peril(self, corpus) -> None:
        # A chipped windscreen and a stolen car are three orders of magnitude apart; a corpus
        # that ignores that trains a model to read the number without reading the sentence.
        glass = [e.label.estimated_amount_brl for e in corpus if e.label.peril == "glass"]
        theft = [e.label.estimated_amount_brl for e in corpus if e.label.peril == "theft"]
        assert max(v for v in glass if v) < min(v for v in theft if v)


class TestNoiseNeverChangesTheTruth:
    """A typo that eats "não" is not noise, it is a different label."""

    def test_protected_tokens_survive_every_noise_setting(self) -> None:
        import random

        sentence = "não houve feridos, nenhum terceiro, três mil e quinhentos reais, sexta passada"
        for seed in range(400):
            noisy = add_noise(random.Random(seed), sentence)
            folded = _fold(noisy).lower()
            for token in ("nao", "nenhum", "tres", "mil", "quinhentos", "sexta", "passada"):
                assert token in folded, f"seed {seed} destroyed {token!r}: {noisy}"

    def test_the_protected_pattern_covers_the_negations(self) -> None:
        for token in ("não", "nao", "ninguém", "nenhuma", "sem", "12", "3.500,00"):
            assert PROTECTED.match(_fold(token)), token

    def test_an_ordinary_word_is_not_protected(self) -> None:
        # If everything were protected, the noise would do nothing and the test above would
        # pass vacuously.
        assert not PROTECTED.match("veiculo")


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


class TestSpelledAmounts:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (3, "três"),
            (15, "quinze"),
            (40, "quarenta"),
            (100, "cem"),
            (350, "trezentos e cinquenta"),
            (1000, "mil"),
            (3500, "três mil e quinhentos"),
            (12000, "doze mil"),
            (45000, "quarenta e cinco mil"),
        ],
    )
    def test_it_says_what_a_person_says(self, value, expected) -> None:
        assert spell_amount(value) == expected

    def test_the_rule_extractor_reads_back_what_the_generator_wrote(self) -> None:
        # The two were written independently from the same language, which is the only reason
        # this is a test rather than a tautology.
        for value in (300, 1500, 3500, 12000, 45000, 90000):
            text = f"orçaram em {spell_amount(value)} reais"
            assert rules.extract_amount(rules.fold(text)) == Decimal(value)


class TestSchema:
    def test_a_fenced_block_is_formatting_and_is_forgiven(self) -> None:
        notice = parse('```json\n{"peril": "theft", "city": "Recife"}\n```')
        assert notice.peril == "theft" and notice.city == "Recife"

    def test_a_trailing_comma_is_forgiven(self) -> None:
        assert parse('{"peril": "fire", "injuries": true,}').injuries is True

    def test_an_invented_peril_is_an_error_not_a_guess(self) -> None:
        with pytest.raises(ParseError, match="peril"):
            parse('{"peril": "earthquake"}')

    def test_a_field_outside_the_schema_is_an_error(self) -> None:
        with pytest.raises(ParseError, match="not in the schema"):
            parse('{"peril": "fire", "driver_name": "someone"}')

    def test_a_state_that_is_not_a_federal_unit_is_an_error(self) -> None:
        with pytest.raises(ParseError, match="federal unit"):
            parse('{"peril": "fire", "state": "XX"}')

    def test_a_brazilian_date_is_understood_and_normalised(self) -> None:
        assert parse('{"peril": "fire", "occurred_on": "12/03/2026"}').occurred_on == date(
            2026, 3, 12
        )

    def test_a_brazilian_amount_is_understood(self) -> None:
        assert parse(
            '{"peril": "fire", "estimated_amount_brl": "R$ 3.500,00"}'
        ).estimated_amount_brl == Decimal("3500.00")

    def test_a_missing_field_is_null_and_not_false(self) -> None:
        notice = parse('{"peril": "fire"}')
        assert notice.injuries is None
        assert notice.injuries is not False

    def test_the_target_string_round_trips(self) -> None:
        original = Notice(
            peril="collision",
            occurred_on=date(2026, 3, 12),
            city="Goiânia",
            state="GO",
            injuries=False,
            estimated_amount_brl=Decimal("3500"),
        )
        assert parse(original.to_target()).to_json() == original.to_json()

    def test_output_with_no_object_at_all_is_a_parse_failure(self) -> None:
        with pytest.raises(ParseError, match="no JSON object"):
            parse("I am sorry, I cannot help with that.")


class TestMetrics:
    def test_the_four_states_are_distinguished(self) -> None:
        assert classify("injuries", None, None) == CORRECT
        assert classify("injuries", None, False) == ASSERTED_ABSENCE
        assert classify("injuries", False, None) == MISSED_VALUE
        assert classify("injuries", False, True) == WRONG

    def test_asserting_false_where_nobody_looked_is_not_merely_wrong(self) -> None:
        """The distinction the whole repository is about.

        A record that says `false` where the message said nothing has put a fact on file that
        no human ever stated, and nothing downstream can tell it from one that was checked.
        """
        assert classify("injuries", None, False) != WRONG

    def test_an_accented_city_matches_its_unaccented_spelling(self) -> None:
        # A third of the corpus has had its accents stripped by the noise model, so a model
        # that reads "Goiania" and writes "Goiania" has read it correctly.
        assert classify("city", "Goiânia", "Goiania") == CORRECT

    def test_a_different_city_does_not_match(self) -> None:
        assert classify("city", "Goiânia", "Brasília") == WRONG

    def test_a_record_that_does_not_parse_is_not_counted_as_nine_wrong_fields(self) -> None:
        truth = Notice(peril="fire")
        result = report("x", [("a", truth)], ["not json at all"])
        assert result.json_validity.successes == 0
        # Nothing was scored field by field, so the field denominator is empty rather than 9.
        assert result.field_accuracy.total == 0
        assert result.exact_record.successes == 0

    def test_asserted_absence_is_measured_against_the_absent_fields_only(self) -> None:
        truth = Notice(peril="fire", injuries=False)
        # Eight fields are absent in the truth; the answer fills one of them.
        answer = '{"peril":"fire","injuries":false,"city":"Recife"}'
        result = report("x", [("a", truth)], [answer])
        assert result.asserted_absence.total == 7
        assert result.asserted_absence.successes == 1


class TestStatistics:
    def test_wilson_matches_a_hand_computed_interval(self) -> None:
        """50 of 100 at 95%: 0.4038 to 0.5962.

        Worth pinning, because the obvious alternatives give different answers and it is easy
        to check one method's number against another's by accident. The exact Clopper-Pearson
        interval for the same data is 0.3983 to 0.6017 — wider, because it guarantees at least
        95% coverage rather than approximately 95%. The normal approximation gives 0.4020 to
        0.5980 and is wrong near the ends, which is why it is not used here.
        """
        interval = wilson(50, 100)
        assert round(interval.low, 4) == 0.4038
        assert round(interval.high, 4) == 0.5962
        assert interval.rate == 0.5

    def test_wilson_stays_inside_zero_and_one_at_the_ends(self) -> None:
        # Where the normal approximation puts the bound below zero, which is exactly where a
        # reader is most likely to believe it.
        assert wilson(0, 30).low == 0.0
        # Exactly 1 in real arithmetic; the last bit is lost to floating point, and clamping it
        # would be fudging the formula to make a test read better.
        assert wilson(30, 30).high == pytest.approx(1.0)

    def test_an_empty_sample_is_not_a_rate_of_zero(self) -> None:
        assert str(wilson(0, 0)) == "n=0"

    def test_the_sign_test_only_counts_disagreements(self) -> None:
        a = [True, True, False, False, True]
        b = [True, False, True, False, True]
        paired = sign_test(a, b)
        assert (paired.better, paired.worse, paired.same) == (1, 1, 3)

    def test_fourteen_against_two_is_significant(self) -> None:
        a = [True] * 2 + [False] * 14
        b = [False] * 2 + [True] * 14
        assert sign_test(a, b).p_value < 0.05

    def test_four_against_two_is_not(self) -> None:
        a = [True] * 2 + [False] * 4
        b = [False] * 2 + [True] * 4
        assert sign_test(a, b).p_value > 0.05

    def test_two_systems_that_never_disagree_are_not_distinguishable(self) -> None:
        assert sign_test([True, False], [True, False]).p_value == 1.0

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same records"):
            sign_test([True], [True, False])


class TestPrompts:
    def test_the_instruction_states_the_rule_about_null(self) -> None:
        assert "null means the message did not say" in prompts.INSTRUCTION
        assert "Never write false" in prompts.INSTRUCTION

    def test_the_instruction_names_every_peril(self) -> None:
        from qlora_lab.schema import PERILS

        for peril in PERILS:
            assert f'"{peril}"' in prompts.SCHEMA_BLOCK

    def test_the_instruction_explains_the_roubo_furto_distinction(self) -> None:
        assert "roubo" in prompts.INSTRUCTION and "furto" in prompts.INSTRUCTION

    def test_the_tuned_prompt_is_a_fraction_of_the_instruction(self) -> None:
        """The claim the serving section makes: ~390 tokens come off the front of every request.

        Measured in characters here because the tokeniser is behind the GPU extra; the ratio is
        what the claim rests on and characters are a faithful proxy for it.
        """
        zero = prompts.zero_shot("bati o carro", date(2026, 3, 12))
        tuned = prompts.tuned("bati o carro", date(2026, 3, 12))
        assert len(tuned) < len(zero) / 8

    def test_the_few_shot_examples_come_only_from_the_pool_it_is_given(self, corpus) -> None:
        train = [e for e in corpus if e.split == "train"]
        shots = prompts.pick_shots(train)
        assert all(s.split == "train" for s in shots)

    def test_the_shots_are_four_different_perils(self, corpus) -> None:
        # A uniform draw regularly gives two of the same, and which two would then be a
        # confound in every comparison the report makes.
        shots = prompts.pick_shots([e for e in corpus if e.split == "train"])
        assert len({s.label.peril for s in shots}) == 4

    def test_the_shots_do_not_move_between_calls(self, corpus) -> None:
        train = [e for e in corpus if e.split == "train"]
        assert [s.id for s in prompts.pick_shots(train)] == [
            s.id for s in prompts.pick_shots(train)
        ]


class TestRuleBaseline:
    """The baseline is held to a floor, not to an exact number.

    An exact number would have to be updated on every harmless change and would stop meaning
    anything. The floor is what the README claims rules can do, and it is the claim CI defends.
    """

    @pytest.fixture(scope="class")
    def validation(self, corpus):
        return [e for e in corpus if e.split == "validation"]

    def test_it_answers_every_record_with_valid_json(self, validation) -> None:
        outputs = [rules.extract(e.text, e.received_on).to_target() for e in validation]
        assert (
            report("rules", [(e.id, e.label) for e in validation], outputs).json_validity.rate
            == 1.0
        )

    def test_it_gets_most_records_exactly_right(self, validation) -> None:
        outputs = [rules.extract(e.text, e.received_on).to_target() for e in validation]
        result = report("rules", [(e.id, e.label) for e in validation], outputs)
        assert result.exact_record.rate > 0.55
        assert result.field_accuracy.rate > 0.90

    def test_it_almost_never_invents_a_value(self, validation) -> None:
        # The thing rules are good at, and the bar the models are measured against.
        outputs = [rules.extract(e.text, e.received_on).to_target() for e in validation]
        result = report("rules", [(e.id, e.label) for e in validation], outputs)
        assert result.asserted_absence.rate < 0.02

    def test_it_does_not_fill_the_state_from_the_gazetteer(self) -> None:
        """It knows Goiânia is in GO. The message did not say so, so the field stays null."""
        notice = rules.extract("bati o carro aqui em Goiânia", date(2026, 3, 12))
        assert notice.city == "Goiânia"
        assert notice.state is None

    def test_it_reads_the_state_when_the_message_gives_it(self) -> None:
        notice = rules.extract("bati o carro em Goiânia, GO", date(2026, 3, 12))
        assert (notice.city, notice.state) == ("Goiânia", "GO")

    def test_the_longer_city_name_wins(self) -> None:
        notice = rules.extract("o carro foi alagado em Aparecida de Goiânia", date(2026, 3, 12))
        assert notice.city == "Aparecida de Goiânia"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("fui roubado e levaram o carro", "robbery"),
            ("me renderam no semáforo", "robbery"),
            ("furtaram meu carro do estacionamento", "theft"),
            ("deixei o carro na rua e quando voltei não estava mais lá", "theft"),
            ("o carro pegou fogo", "fire"),
            ("trincou o para-brisa", "glass"),
            ("atropelei um animal na pista", "animal"),
            ("riscaram a lataria toda", "vandalism"),
            ("bati na traseira de outro carro", "collision"),
        ],
    )
    def test_it_separates_roubo_from_furto(self, text, expected) -> None:
        assert rules.extract(text, date(2026, 3, 12)).peril == expected

    def test_a_negation_applies_to_its_own_clause_only(self) -> None:
        """ "Não houve feridos, o carro está rodando" is one negation and two facts."""
        notice = rules.extract(
            "bati o carro, não houve feridos, o carro ainda está rodando normalmente",
            date(2026, 3, 12),
        )
        assert notice.injuries is False
        assert notice.vehicle_drivable is True

    def test_a_tow_means_not_drivable_without_any_negation(self) -> None:
        notice = rules.extract("bati o carro, tive que chamar guincho", date(2026, 3, 12))
        assert notice.vehicle_drivable is False

    def test_sexta_passada_is_the_friday_before_not_today(self) -> None:
        # 2026-03-13 is a Friday. "Sexta passada" said on a Friday means the previous one.
        assert rules.extract("bati o carro na sexta passada", date(2026, 3, 13)).occurred_on == (
            date(2026, 3, 6)
        )

    def test_a_date_without_a_year_does_not_land_in_the_future(self) -> None:
        # Said on 10 January about "dia 28/12", the loss is last December, not this one.
        assert rules.extract("bati o carro no dia 28/12", date(2026, 1, 10)).occurred_on == (
            date(2025, 12, 28)
        )


#: Written down rather than computed at import, which would make the test assert only that
#: the generator equals itself.
FROZEN_TEST_DIGEST = "caf72765ebd5d3c16d8645ca48bf171f2659351db474f718daf7eebed5cd9af8"


class TestTheShippedData:
    """The committed splits must be the ones the generator produces.

    They are committed so the repository can be read, and so a reviewer without a GPU can run
    the rule baseline and reproduce one row of the table. A committed file that has drifted from
    its generator is worse than no file at all.
    """

    def test_the_committed_splits_match_the_generator(self, corpus) -> None:
        directory = Path(__file__).resolve().parent.parent / "data"
        if not directory.exists():
            pytest.skip("run `python -m qlora_lab data` first")
        from qlora_lab.generate import read

        for split, _ in SPLITS:
            committed = read(directory, split)
            expected = [e for e in corpus if e.split == split]
            assert [c.text for c in committed] == [e.text for e in expected]
            assert [c.label.to_target() for c in committed] == [
                e.label.to_target() for e in expected
            ]

    def test_no_field_was_added_without_the_report_knowing(self) -> None:
        # `FIELDS` drives the prompt, the training target and every per-field row in the report.
        assert len(FIELDS) == 9
        assert FIELDS[0] == "peril"
