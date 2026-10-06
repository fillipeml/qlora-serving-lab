"""A seeded generator for Portuguese first-notice-of-loss messages, with exact labels.

**Why generated and not collected.** A real notice of loss is somebody's accident: a date, a
place, a vehicle and often an injury. There is no version of that corpus that can be committed
to a public repository. Generating it is not a compromise here, it is the better instrument: the
record is drawn first and the message is written from it, so the label is exact by construction
rather than by annotation, and — the part that matters for this repository — the generator knows
which fields it deliberately withheld. That is the only way to measure asserted absence, which
turns out to be where the models differ most.

**The noise is the point.** Brazilian Portuguese typed on a phone loses its accents, gains
typos, writes amounts in words, and says "sexta passada" instead of a date. Every one of those
is applied after the record exists, and never to a token that carries meaning: a typo that eats
the "não" in "não houve feridos" would silently change the truth, so negations, numerals,
weekday names and place names are protected. `test_noise_never_changes_the_truth` asserts it.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from .schema import Notice, from_json

#: The window the notices arrive in. Fixed so the resolved dates are stable across runs.
FIRST_DAY = date(2026, 1, 6)
LAST_DAY = date(2026, 9, 28)

WEEKDAYS = ("segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo")

CITIES: tuple[tuple[str, str], ...] = (
    ("Goiânia", "GO"),
    ("Anápolis", "GO"),
    ("Aparecida de Goiânia", "GO"),
    ("Brasília", "DF"),
    ("Belo Horizonte", "MG"),
    ("Uberlândia", "MG"),
    ("São Paulo", "SP"),
    ("Campinas", "SP"),
    ("Santo André", "SP"),
    ("Rio de Janeiro", "RJ"),
    ("Niterói", "RJ"),
    ("Curitiba", "PR"),
    ("Londrina", "PR"),
    ("Porto Alegre", "RS"),
    ("Caxias do Sul", "RS"),
    ("Florianópolis", "SC"),
    ("Joinville", "SC"),
    ("Salvador", "BA"),
    ("Feira de Santana", "BA"),
    ("Recife", "PE"),
    ("Fortaleza", "CE"),
    ("Manaus", "AM"),
    ("Belém", "PA"),
    ("Cuiabá", "MT"),
    ("Campo Grande", "MS"),
    ("Vitória", "ES"),
    ("Natal", "RN"),
)

#: How the message opens. One template per peril, and the peril is the one field never withheld
#: — a message that does not say what happened is not a notice of loss.
OPENINGS: dict[str, tuple[str, ...]] = {
    "collision": (
        "bati o carro",
        "sofri uma colisão com outro veículo",
        "fui atingido na traseira enquanto estava parado",
        "colidi na lateral de outro carro ao mudar de faixa",
        "entrei em um engavetamento",
        "bati em um poste ao desviar de um buraco",
    ),
    "robbery": (
        "fui roubado e levaram o carro",
        "me abordaram armados e tomaram o veículo",
        "sofri um roubo à mão armada e perdi o carro",
        "dois homens me renderam no semáforo e levaram o carro",
    ),
    "theft": (
        "furtaram meu carro do estacionamento",
        "deixei o carro na rua e quando voltei não estava mais lá",
        "levaram o veículo da garagem sem que eu percebesse",
        "sofri um furto, sumiram com o carro durante a madrugada",
    ),
    "fire": (
        "o carro pegou fogo",
        "houve um incêndio no compartimento do motor",
        "saiu fumaça do painel e o veículo incendiou",
        "o veículo foi consumido pelo fogo depois de uma pane elétrica",
    ),
    "flood": (
        "o carro foi alagado",
        "a água subiu na enchente e cobriu o veículo",
        "fiquei preso em um alagamento e a água entrou no carro",
        "o rio transbordou e o carro ficou submerso",
    ),
    "glass": (
        "trincou o para-brisa",
        "uma pedra da pista quebrou o vidro dianteiro",
        "quebraram o vidro lateral do carro",
        "o para-brisa estourou sozinho com a variação de temperatura",
    ),
    "animal": (
        "atropelei um animal na pista",
        "um cachorro atravessou na frente e eu bati nele",
        "um boi invadiu a rodovia e colidi com o animal",
        "bati em uma capivara que cruzou a via",
    ),
    "vandalism": (
        "riscaram a lataria toda do carro",
        "depredaram o veículo no estacionamento",
        "vandalizaram o carro durante a noite, amassaram as portas",
        "chutaram o retrovisor e riscaram a pintura de propósito",
    ),
}

GREETINGS = ("bom dia", "boa tarde", "boa noite", "olá", "prezados")
PREAMBLES = (
    "preciso abrir um sinistro",
    "venho comunicar um sinistro",
    "gostaria de registrar uma ocorrência",
    "estou comunicando o ocorrido",
)
CLOSINGS = (
    "aguardo retorno",
    "obrigado",
    "fico no aguardo das orientações",
    "por favor me informem os próximos passos",
    "desde já agradeço",
)

THIRD_PARTY_TRUE = (
    "tinha outro veículo envolvido",
    "o outro motorista parou e trocamos os dados",
    "havia um terceiro envolvido na ocorrência",
)
THIRD_PARTY_FALSE = (
    "não teve outro veículo envolvido",
    "foi só o meu carro, sem terceiros",
    "nenhum terceiro se envolveu",
)
INJURIES_TRUE = (
    "minha passageira se machucou e foi levada ao hospital",
    "tive um corte no braço",
    "houve feridos",
    "o outro motorista ficou ferido",
)
INJURIES_FALSE = (
    "ninguém se feriu",
    "graças a Deus não houve feridos",
    "saímos todos sem nenhum arranhão",
)
REPORT_TRUE = (
    "já registrei o boletim de ocorrência",
    "fiz o BO na delegacia no mesmo dia",
    "o boletim de ocorrência está registrado",
)
REPORT_FALSE = (
    "não cheguei a fazer boletim de ocorrência",
    "ainda não registrei BO nenhum",
    "não fizemos boletim",
)
DRIVABLE_TRUE = (
    "o carro ainda está rodando normalmente",
    "dá para dirigir, só ficou o dano na lataria",
    "consegui levar o veículo para casa por conta própria",
)
DRIVABLE_FALSE = (
    "o carro não sai do lugar",
    "tive que chamar guincho, o veículo não anda",
    "o veículo ficou imobilizado no local",
)

#: Which fields a given peril can truthfully carry, and in which direction.
#:
#: Without this the generator writes a vandalism claim with another vehicle involved, or a
#: cracked windscreen that immobilised the car. Every such record teaches a model something
#: false and scores it on something nobody would ever write. The asymmetry is deliberate: a
#: person reporting a fire may well say "não houve terceiros envolvidos", but will not report a
#: third party colliding with them; so the negative clause is available everywhere and only the
#: positive one is restricted.
THIRD_PARTY_POSSIBLE = frozenset({"collision", "animal"})
INJURIES_POSSIBLE = frozenset({"collision", "animal", "fire", "flood", "robbery"})
#: A stolen car is not in the owner's hands to be drivable or not.
DRIVABLE_APPLIES = frozenset({"collision", "animal", "fire", "flood", "glass", "vandalism"})

#: What the loss is worth, by peril. A total loss and a chipped windscreen are three orders of
#: magnitude apart, and a corpus that ignores that trains a model to read the number without
#: reading the sentence.
AMOUNT_RANGE: dict[str, tuple[int, int]] = {
    "glass": (300, 2_500),
    "vandalism": (600, 9_000),
    "animal": (1_200, 18_000),
    "collision": (900, 40_000),
    "flood": (4_000, 60_000),
    "fire": (12_000, 90_000),
    "theft": (18_000, 95_000),
    "robbery": (18_000, 95_000),
}

UNITS = (
    "zero",
    "um",
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
    "dez",
    "onze",
    "doze",
    "treze",
    "quatorze",
    "quinze",
    "dezesseis",
    "dezessete",
    "dezoito",
    "dezenove",
)
TENS = (
    "",
    "",
    "vinte",
    "trinta",
    "quarenta",
    "cinquenta",
    "sessenta",
    "setenta",
    "oitenta",
    "noventa",
)
HUNDREDS = (
    "",
    "cento",
    "duzentos",
    "trezentos",
    "quatrocentos",
    "quinhentos",
    "seiscentos",
    "setecentos",
    "oitocentos",
    "novecentos",
)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


#: Tokens that carry truth. Noise never touches them, because a typo in "não" or in a digit is
#: not noise — it is a different label. Everything else is fair game.
#:
#: Built from accent-stripped forms and matched against accent-stripped text. The first version
#: of this was built from the words as written — "três", "sábado" — and matched against text the
#: noise model had already stripped, so every accented numeral and weekday was unprotected and
#: silently corruptible. `test_protected_tokens_survive_every_noise_setting` found it at seed 29.
_PROTECTED_WORDS = (
    "nao",
    "nenhum",
    "nenhuma",
    "ninguem",
    "sem",
    "mil",
    "reais",
    "ontem",
    "anteontem",
    "hoje",
    "cem",
    *(_strip_accents(w) for w in UNITS),
    *(_strip_accents(w) for w in TENS if w),
    *(_strip_accents(w) for w in HUNDREDS if w),
    *(_strip_accents(w) for w in WEEKDAYS),
)

PROTECTED = re.compile(
    r"^(?:"
    + "|".join(sorted(set(_PROTECTED_WORDS), key=len, reverse=True))
    + r"|passad[ao]|ultim[ao]|\d[\d.,/:-]*"
    + r")$",
    re.IGNORECASE,
)


def spell_amount(value: int) -> str:
    """An integer number of reais, written the way somebody says it out loud.

    Needed because a notice dictated over the phone or typed in a hurry says "três mil e
    quinhentos", and an extractor that only reads digits scores perfectly on a corpus that only
    contains digits. Repository 3 feeds this one from a transcript, where the digits are gone
    entirely.
    """

    def under_thousand(n: int) -> str:
        if n == 100:
            return "cem"
        parts = []
        if n >= 100:
            parts.append(HUNDREDS[n // 100])
            n %= 100
        if n >= 20:
            parts.append(TENS[n // 10])
            n %= 10
            if n:
                parts.append(UNITS[n])
        elif n:
            parts.append(UNITS[n])
        return " e ".join(parts)

    if value < 1000:
        return under_thousand(value)
    thousands, rest = divmod(value, 1000)
    head = "mil" if thousands == 1 else f"{under_thousand(thousands)} mil"
    if not rest:
        return head
    # "três mil e quinhentos", but "três mil, cento e vinte" — the conjunction is only used when
    # the remainder is round or under a hundred, which is how it is actually spoken.
    joiner = " e " if rest < 100 or rest % 100 == 0 else ", "
    return f"{head}{joiner}{under_thousand(rest)}"


@dataclass(frozen=True)
class Example:
    """One message, the record it was written from, and the date the insurer received it."""

    id: str
    split: str
    received_on: date
    text: str
    label: Notice
    #: Which clauses were deliberately withheld. Carried so a failure can be read without
    #: re-deriving it from the record.
    withheld: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "split": self.split,
            "received_on": self.received_on.isoformat(),
            "text": self.text,
            "label": self.label.to_json(),
            "withheld": list(self.withheld),
        }


def _date_clause(rng: random.Random, received: date) -> tuple[str, date]:
    """A way of saying when, and the date it resolves to.

    Relative forms are the majority on purpose. "Sexta passada" is one subtraction for a person
    and a genuine inference for a model, and it is the field where the three systems in the
    report separate most sharply.
    """
    style = rng.choices(
        ("today", "yesterday", "ereyesterday", "weekday", "days_ago", "numeric", "written"),
        weights=(8, 16, 8, 20, 14, 20, 14),
    )[0]
    if style == "today":
        return rng.choice(("hoje de manhã", "hoje cedo", "hoje")), received
    if style == "yesterday":
        return rng.choice(("ontem", "ontem à tarde", "ontem de manhã")), received - timedelta(1)
    if style == "ereyesterday":
        return "anteontem", received - timedelta(2)
    if style == "weekday":
        target = rng.randrange(7)
        # The most recent occurrence strictly before the day it was received: "sexta passada"
        # said on a Friday means the previous Friday, not today.
        delta = (received.weekday() - target) % 7 or 7
        when = received - timedelta(delta)
        name = WEEKDAYS[target]
        return rng.choice((f"na {name} passada", f"{name} passada", f"na última {name}")), when
    if style == "days_ago":
        n = rng.randrange(3, 12)
        return rng.choice((f"há {n} dias", f"faz {n} dias")), received - timedelta(n)
    n = rng.randrange(1, 25)
    when = received - timedelta(n)
    if style == "numeric":
        form = rng.choice(
            (
                f"no dia {when.day:02d}/{when.month:02d}",
                f"em {when.day:02d}/{when.month:02d}/{when.year}",
            )
        )
        return form, when
    months = (
        "janeiro",
        "fevereiro",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    )
    return f"no dia {when.day} de {months[when.month - 1]}", when


def _amount_clause(rng: random.Random, peril: str) -> tuple[str, Decimal]:
    low, high = AMOUNT_RANGE[peril]
    # Rounded the way a quote is quoted: to the hundred below ten thousand, to the thousand
    # above it. Nobody says "o orçamento ficou em R$ 37.412,00".
    if high <= 10_000:
        value = rng.randrange(low // 100, high // 100 + 1) * 100
    else:
        value = rng.randrange(max(low // 1000, 1), high // 1000 + 1) * 1000
    style = rng.choices(("formal", "loose", "spelled"), weights=(10, 6, 5))[0]
    if style == "formal":
        text = f"R$ {value:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")
        clause = rng.choice((f"o orçamento ficou em {text}", f"o prejuízo é de {text}"))
    elif style == "loose":
        # "uns 5 mil" only when the value really is five thousand. The first version wrote
        # `value // 1000` unconditionally, so a loss of 5,900 was reported as "uns 5 mil" while
        # the label said 5900 — a message stating an amount the label contradicts. 37 of 772
        # records carried it, and every system was scored wrong on all of them for reading the
        # text correctly. `test_every_stated_amount_is_recoverable_from_the_text` is the guard.
        #
        # One rng.choice either way, so the fix changes those records' wording and nothing else
        # about the corpus.
        thousands = value >= 1000 and value % 1000 == 0
        clause = rng.choice(
            (
                f"orçaram em uns {value // 1000} mil reais"
                if thousands
                else f"orçaram em uns {value} reais",
                f"o conserto sai por {value // 1000} mil"
                if thousands
                else f"o conserto sai por {value} reais",
            )
        )
    else:
        clause = rng.choice(
            (
                f"orçaram em {spell_amount(value)} reais",
                f"me falaram em {spell_amount(value)} reais de conserto",
            )
        )
    return clause, Decimal(value)


def _place_clause(rng: random.Random) -> tuple[str, str, str | None]:
    city, uf = rng.choice(CITIES)
    if rng.random() < 0.45:
        return rng.choice((f"em {city}, {uf}", f"aqui em {city}/{uf}")), city, uf
    return rng.choice((f"aqui em {city}", f"em {city}", f"na cidade de {city}")), city, None


def _typo(rng: random.Random, word: str) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.random()
    if kind < 0.4:
        return word[:i] + word[i + 1 :]
    if kind < 0.7:
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    return word[:i] + word[i] + word[i:]


def add_noise(rng: random.Random, text: str) -> str:
    """Accents dropped, the odd typo, and the casing people actually use.

    Applied token by token so a protected token can be skipped. The accent-stripping is applied
    to the whole message because that is how it happens — a phone keyboard without accents loses
    all of them, not a random third.
    """
    if rng.random() < 0.35:
        text = _strip_accents(text)
    if rng.random() < 0.10:
        text = text.upper()
    elif rng.random() < 0.12:
        text = text.lower()
    rate = rng.choice((0.0, 0.0, 0.02, 0.05))
    if not rate:
        return text
    out = []
    for token in text.split(" "):
        core = token.strip(".,;:!?")
        if core and not PROTECTED.match(_strip_accents(core)) and rng.random() < rate:
            out.append(token.replace(core, _typo(rng, core), 1))
        else:
            out.append(token)
    return " ".join(out)


#: How often each optional field is stated at all. Tuned so roughly a third of every record is
#: absent: enough that asserted absence is measurable on 300 test records, and close to what a
#: free-text notice actually contains.
PRESENCE: dict[str, float] = {
    "occurred_on": 0.88,
    "place": 0.70,
    "third_party_involved": 0.55,
    "injuries": 0.62,
    "police_report_filed": 0.50,
    "vehicle_drivable": 0.48,
    "estimated_amount_brl": 0.52,
}


def _example(rng: random.Random, index: int, split: str) -> Example:
    received = FIRST_DAY + timedelta(rng.randrange((LAST_DAY - FIRST_DAY).days + 1))
    peril = rng.choice(tuple(OPENINGS))
    opening = rng.choice(OPENINGS[peril])

    withheld: list[str] = []
    clauses: list[str] = []
    occurred_on = city = state = amount = None

    if rng.random() < PRESENCE["occurred_on"]:
        clause, occurred_on = _date_clause(rng, received)
        clauses.append(clause)
    else:
        withheld.append("occurred_on")

    if rng.random() < PRESENCE["place"]:
        clause, city, state = _place_clause(rng)
        clauses.append(clause)
    else:
        withheld.append("city")
    if state is None:
        withheld.append("state")

    booleans: dict[str, bool | None] = {}
    for field, (yes, no), applies, positive in (
        ("third_party_involved", (THIRD_PARTY_TRUE, THIRD_PARTY_FALSE), None, THIRD_PARTY_POSSIBLE),
        ("injuries", (INJURIES_TRUE, INJURIES_FALSE), None, INJURIES_POSSIBLE),
        ("police_report_filed", (REPORT_TRUE, REPORT_FALSE), None, None),
        ("vehicle_drivable", (DRIVABLE_TRUE, DRIVABLE_FALSE), DRIVABLE_APPLIES, None),
    ):
        stateable = applies is None or peril in applies
        if stateable and rng.random() < PRESENCE[field]:
            value = rng.random() < 0.55 and (positive is None or peril in positive)
            booleans[field] = value
            clauses.append(rng.choice(yes if value else no))
        else:
            booleans[field] = None
            withheld.append(field)

    if rng.random() < PRESENCE["estimated_amount_brl"]:
        clause, amount = _amount_clause(rng, peril)
        clauses.append(clause)
    else:
        withheld.append("estimated_amount_brl")

    # The opening always leads; everything after it is in the order it occurred to the person
    # writing, which is to say shuffled.
    rng.shuffle(clauses)
    body = ", ".join([opening, *clauses])

    parts = []
    if rng.random() < 0.55:
        parts.append(rng.choice(GREETINGS).capitalize())
    if rng.random() < 0.50:
        parts.append(rng.choice(PREAMBLES))
    parts.append(body)
    if rng.random() < 0.45:
        parts.append(rng.choice(CLOSINGS))
    text = ". ".join(parts) + "."
    text = add_noise(rng, text)

    return Example(
        id=f"FNOL-{index:05d}",
        split=split,
        received_on=received,
        text=text,
        label=Notice(
            peril=peril,
            occurred_on=occurred_on,
            city=city,
            state=state,
            estimated_amount_brl=amount,
            **booleans,
        ),
        withheld=tuple(withheld),
    )


#: Sizes chosen from what the two halves need. 1,000 training examples is more than a LoRA on a
#: 0.6B model needs to learn a format and less than an evening of compute on one consumer card;
#: 300 test records put the 95% interval on a record-level rate at roughly ±5 points, which is
#: narrow enough to separate the systems the report compares.
SPLITS: tuple[tuple[str, int], ...] = (("train", 1000), ("validation", 200), ("test", 300))


def generate(seed: int = 20261005) -> list[Example]:
    """Every example, deterministically.

    The split is decided before the text is written rather than by shuffling afterwards, so
    adding a split never renumbers the others, and the test set is the same 300 records in every
    run of this repository's history.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[Example] = []
    index = 0
    for split, count in SPLITS:
        made = 0
        while made < count:
            candidate = _example(rng, index, split)
            index += 1
            if candidate.text in seen:
                continue
            seen.add(candidate.text)
            out.append(candidate)
            made += 1
    return out


def write(directory: Path, examples: list[Example]) -> dict[str, int]:
    directory.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    names = {e.split for e in examples}
    for split, _ in SPLITS + tuple((n, 0) for n in sorted(names - {s for s, _ in SPLITS})):
        rows = [e for e in examples if e.split == split]
        path = directory / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(e.to_json(), ensure_ascii=False) + "\n" for e in rows),
            encoding="utf-8",
        )
        counts[split] = len(rows)
    return counts


def read(directory: Path, split: str) -> list[Example]:
    path = directory / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        Example(
            id=r["id"],
            split=r["split"],
            received_on=date.fromisoformat(r["received_on"]),
            text=r["text"],
            label=from_json(r["label"]),
            withheld=tuple(r["withheld"]),
        )
        for r in rows
    ]
