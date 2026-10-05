"""The baseline nobody runs: no model at all.

Every comparison of language models to each other quietly assumes a model is needed. For nine
fields of a short, conventional message in one language, that assumption deserves a number. This
module is that number — regular expressions, a municipality gazetteer and a date parser, in
about three hundred lines and roughly a millisecond per record.

**Read its score as a ceiling, not a baseline.** The same person wrote this and wrote the
generator. The cues it looks for are the cues the generator emits, and no production rule set
ever has that. So it measures the best rules could possibly do against a known distribution, and
a model that merely matches it has not yet earned its GPU; a model that beats it has beaten
rules under conditions that favour rules enormously.

The one thing it cannot do is the thing rules never can: it has no way to decide that a sentence
it does not recognise means something. The report shows that as missed values rather than wrong
ones, which is the correct failure for a rule system to have.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from decimal import Decimal

from .generate import CITIES, HUNDREDS, TENS, UNITS
from .schema import Notice

#: Municipality, federal unit. A real deployment would carry all 5,570; the shape of the lookup
#: is the same and the gazetteer is public data either way.
GAZETTEER: dict[str, tuple[str, str]] = {}

WEEKDAY_NAMES = {
    "segunda": 0,
    "terca": 1,
    "quarta": 2,
    "quinta": 3,
    "sexta": 4,
    "sabado": 5,
    "domingo": 6,
}
MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

#: Cues for each peril, most specific first. Order matters: "levaram o carro" appears in both a
#: robbery and a theft, and only the violence words separate them, so those are tested first.
PERIL_CUES: tuple[tuple[str, str], ...] = (
    ("robbery", r"roub|rendera|rendeu|armad|mao armada|sob ameaca|abordara.{0,30}arma"),
    ("theft", r"furt|levaram o veiculo|sumiram com|nao estava mais la|sem que eu percebesse"),
    ("fire", r"incendi|pegou fogo|consumido pelo fogo|fuma[cç]a.{0,20}painel"),
    ("flood", r"alag|enchente|submers|agua (subiu|entrou)|transbordou"),
    ("glass", r"para-?brisa|vidro (lateral|dianteiro|traseiro)|trincou|estourou sozinho"),
    ("animal", r"animal|cachorro|boi|capivara|atropelei"),
    ("vandalism", r"vandal|depredar|depredaram|riscaram|chutaram|de proposito|amassaram as portas"),
    ("collision", r"bati|colid|colis|engavetamento|atingido"),
)

NEGATION = r"(nao|nenhum|nenhuma|ninguem|sem|nunca)"

BOOLEAN_CUES: dict[str, tuple[str, str]] = {
    # (cue for the subject, extra positive-only cue). The sign comes from whether a negation
    # sits within the clause, which is why the clause boundary matters more than the keyword.
    "third_party_involved": (r"terceiro|outro (veiculo|motorista|carro)|outra pessoa", ""),
    "injuries": (r"ferid|machuc|lesion|hospital|corte no|arranhao", ""),
    "police_report_filed": (
        r"\bbo\b|boletim de ocorrencia|delegacia|registrei.{0,20}ocorrencia",
        "",
    ),
    "vehicle_drivable": (
        r"rodando|dirigir|guincho|imobilizad|nao (sai|anda)|dá para rodar|da para rodar",
        "",
    ),
}

#: Within `vehicle_drivable`, some cues are themselves negative regardless of a "não": a car
#: that needed a tow is not drivable, and the sentence saying so contains no negation at all.
INHERENTLY_FALSE = re.compile(r"guincho|imobilizad|nao sai do lugar|nao anda|nao roda")
INHERENTLY_TRUE = re.compile(r"rodando normal|da para dirigir|da para rodar|por conta propria")


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    )


def _build_gazetteer() -> None:
    for city, uf in CITIES:
        GAZETTEER[fold(city)] = (city, uf)


_build_gazetteer()

_WORD_VALUES: dict[str, int] = {}
for _i, _w in enumerate(UNITS):
    _WORD_VALUES[fold(_w)] = _i
for _i, _w in enumerate(TENS):
    if _w:
        _WORD_VALUES[fold(_w)] = _i * 10
for _i, _w in enumerate(HUNDREDS):
    if _w:
        _WORD_VALUES[fold(_w)] = _i * 100
_WORD_VALUES["cem"] = 100


def spelled_number(tokens: list[str]) -> int | None:
    """Read "três mil e quinhentos" back into 3500.

    Left to right, accumulating below a thousand and multiplying out when "mil" arrives. It
    stops at the first token it does not know, which is what makes it usable on free text: the
    number ends where the sentence resumes.
    """
    total = 0
    current = 0
    consumed = False
    for raw in tokens:
        token = fold(raw.strip(".,;:!?"))
        if token == "e":
            continue
        if token == "mil":
            current = current or 1
            total += current * 1000
            current = 0
            consumed = True
            continue
        if token in _WORD_VALUES:
            current += _WORD_VALUES[token]
            consumed = True
            continue
        break
    if not consumed:
        return None
    return total + current


def extract_peril(folded: str) -> str:
    for peril, pattern in PERIL_CUES:
        if re.search(pattern, folded):
            return peril
    # Something must be returned: the schema has no null peril, and a rule set that declines to
    # answer would be scored as unparseable rather than as wrong. Collision is the modal peril
    # in any motor book, so guessing it is the least informative guess available.
    return "collision"


def extract_date(folded: str, received: date) -> date | None:
    if re.search(r"\bhoje\b", folded):
        return received
    if re.search(r"\banteontem\b", folded):
        return received - timedelta(2)
    if re.search(r"\bontem\b", folded):
        return received - timedelta(1)
    ago = re.search(r"(?:ha|faz)\s+(\d+)\s+dias", folded)
    if ago:
        return received - timedelta(int(ago.group(1)))
    weekday = re.search(
        r"(?:na\s+)?(?:ultima\s+)?(" + "|".join(WEEKDAY_NAMES) + r")(?:\s+passad[ao])?", folded
    )
    if weekday and re.search(r"passad[ao]|ultim[ao]", folded):
        target = WEEKDAY_NAMES[weekday.group(1)]
        delta = (received.weekday() - target) % 7 or 7
        return received - timedelta(delta)
    full = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", folded)
    if full:
        day, month, year = (int(g) for g in full.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None
    short = re.search(r"\bdia\s+(\d{1,2})/(\d{1,2})\b", folded)
    if short:
        day, month = (int(g) for g in short.groups())
        # No year in the text. The notice is about something that has already happened, so the
        # year is the received year unless that would put the loss in the future.
        for year in (received.year, received.year - 1):
            try:
                candidate = date(year, month, day)
            except ValueError:
                continue
            if candidate <= received:
                return candidate
        return None
    written = re.search(r"\bdia\s+(\d{1,2})\s+de\s+(" + "|".join(MONTHS) + r")", folded)
    if written:
        day = int(written.group(1))
        month = MONTHS[written.group(2)]
        for year in (received.year, received.year - 1):
            try:
                candidate = date(year, month, day)
            except ValueError:
                continue
            if candidate <= received:
                return candidate
    return None


def extract_place(text: str) -> tuple[str | None, str | None]:
    folded = fold(text)
    # Longest name first, so "Aparecida de Goiânia" is not shortened to "Goiânia".
    for key in sorted(GAZETTEER, key=len, reverse=True):
        position = folded.find(key)
        if position < 0:
            continue
        city, default_uf = GAZETTEER[key]
        tail = folded[position + len(key) : position + len(key) + 6]
        explicit = re.match(r"\s*[,/]\s*([a-z]{2})\b", tail)
        if explicit and explicit.group(1).upper() == default_uf:
            return city, default_uf
        # The gazetteer knows the federal unit, but the message did not state it, and this
        # field records what the message said. Filling it from the gazetteer would be exactly
        # the asserted absence the whole report is about.
        return city, None
    return None, None


def extract_amount(folded: str) -> Decimal | None:
    formal = re.search(r"r\$\s*([\d.]+,\d{2}|\d[\d.]*)", folded)
    if formal:
        digits = formal.group(1)
        if "," in digits:
            digits = digits.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", digits):
            digits = digits.replace(".", "")
        return Decimal(digits)
    loose = re.search(r"(\d+)\s*mil\b", folded)
    if loose:
        return Decimal(loose.group(1)) * 1000
    plain = re.search(r"\b(\d{3,6})\s*reais\b", folded)
    if plain:
        return Decimal(plain.group(1))
    spelled = re.search(r"(?:em|por|de|uns)\s+((?:[a-z]+\s+){0,6}?)(?:reais|mil)", folded)
    if spelled:
        start = spelled.start(1)
        value = spelled_number(folded[start:].split())
        if value:
            return Decimal(value)
    return None


def _clauses(folded: str) -> list[str]:
    """Split on the punctuation the messages actually use.

    The clause is the unit a negation applies to. "Não houve feridos, o carro está rodando"
    carries one negation and two facts, and a document-level negation check would make the
    second one false.
    """
    return [c.strip() for c in re.split(r"[,.;!?]", folded) if c.strip()]


def extract_boolean(folded: str, field: str) -> bool | None:
    pattern, _ = BOOLEAN_CUES[field]
    for clause in _clauses(folded):
        if not re.search(pattern, clause):
            continue
        if field == "vehicle_drivable":
            if INHERENTLY_FALSE.search(clause):
                return False
            if INHERENTLY_TRUE.search(clause):
                return True
        negated = re.search(NEGATION, clause) is not None
        return not negated
    return None


def extract(text: str, received_on: date) -> Notice:
    """One record, from the text and the date the insurer received it."""
    folded = fold(text)
    city, state = extract_place(text)
    return Notice(
        peril=extract_peril(folded),
        occurred_on=extract_date(folded, received_on),
        city=city,
        state=state,
        third_party_involved=extract_boolean(folded, "third_party_involved"),
        injuries=extract_boolean(folded, "injuries"),
        police_report_filed=extract_boolean(folded, "police_report_filed"),
        vehicle_drivable=extract_boolean(folded, "vehicle_drivable"),
        estimated_amount_brl=extract_amount(folded),
    )
