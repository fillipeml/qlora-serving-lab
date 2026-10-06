"""A second test set, written in words none of the systems have seen.

**Why this exists.** The rule baseline and the corpus generator were written by the same person,
so the rules look for exactly the cues the generator emits. That makes the rules' score a
ceiling on what rules can do against a distribution they already know — which is not a situation
any production system is ever in. The fine-tuned model has the matching problem from the other
direction: it was trained on a thousand messages drawn from the same templates as the test set.

Both numbers are therefore optimistic, and they are optimistic in ways that do not cancel.

This module writes 200 further records using a **held-out vocabulary**: different words for the
same eight perils, different ways of giving a date, different phrasings for every boolean, and
twenty-four municipalities that are not in the rule set's gazetteer. The records are drawn the
same way and the labels are exact by construction, exactly as before. Nothing else changes.

The words were chosen to be ordinary Portuguese that any adult speaker reads without effort, and
specifically *not* synonyms of the training vocabulary. `assalto` is the commonest Brazilian word
for a robbery and the rule set does not contain it. `rebocado` is what a tow is called and the
rule set looks for `guincho`. `vítimas` is what a police report says and the rule set looks for
`feridos`. None of those is a trick; each is the other obvious way to say the same thing, and
whether a system survives it is the only question in this repository that bears on production.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

from .generate import (
    AMOUNT_RANGE,
    DRIVABLE_APPLIES,
    INJURIES_POSSIBLE,
    PRESENCE,
    THIRD_PARTY_POSSIBLE,
    Example,
    add_noise,
    spell_amount,
)
from .schema import Notice

#: Municipalities deliberately absent from `rules.GAZETTEER`. All real, none in the training
#: corpus: a rule set with a 27-city lookup returns null for every one of them, and a model that
#: has read any Portuguese at all should recognise them as place names.
HELD_OUT_CITIES: tuple[tuple[str, str], ...] = (
    ("Sorocaba", "SP"),
    ("Ribeirão Preto", "SP"),
    ("Bauru", "SP"),
    ("Araçatuba", "SP"),
    ("Juiz de Fora", "MG"),
    ("Montes Claros", "MG"),
    ("Governador Valadares", "MG"),
    ("Petrolina", "PE"),
    ("Caruaru", "PE"),
    ("Marabá", "PA"),
    ("Santarém", "PA"),
    ("Chapecó", "SC"),
    ("Criciúma", "SC"),
    ("Passo Fundo", "RS"),
    ("Pelotas", "RS"),
    ("Maringá", "PR"),
    ("Cascavel", "PR"),
    ("Imperatriz", "MA"),
    ("Mossoró", "RN"),
    ("Juazeiro do Norte", "CE"),
    ("Rio Branco", "AC"),
    ("Porto Velho", "RO"),
    ("Palmas", "TO"),
    ("Macapá", "AP"),
)

#: The same eight perils, said differently. `assalto`, `subtração`, `abalroamento`, `semovente`
#: and `combustão` are all ordinary Portuguese and none appears in the rule set's cue patterns.
HELD_OUT_OPENINGS: dict[str, tuple[str, ...]] = {
    "collision": (
        "me envolvi em um acidente de trânsito",
        "houve um abalroamento na via",
        "meu veículo foi abalroado por outro condutor",
        "aconteceu um choque entre o meu automóvel e outro",
    ),
    "robbery": (
        "sofri um assalto e perdi o automóvel",
        "fui vítima de assalto à mão armada e perdi o carro",
        "tomaram meu automóvel mediante grave ameaça",
        "fui abordado por assaltantes que ficaram com o veículo",
    ),
    "theft": (
        "subtraíram meu automóvel da garagem",
        "fui vítima de subtração do veículo",
        "meu carro desapareceu da vaga onde eu havia estacionado",
        "o automóvel sumiu do pátio durante a madrugada",
    ),
    "fire": (
        "houve combustão no automóvel",
        "as chamas tomaram conta do veículo",
        "o automóvel foi destruído pelas chamas após uma falha elétrica",
        "princípio de incêndio que consumiu o automóvel",
    ),
    "flood": (
        "a correnteza invadiu a garagem e levou o automóvel",
        "meu automóvel ficou debaixo d'água",
        "a maré de chuva encobriu o veículo por completo",
        "a inundação deixou o automóvel submerso",
    ),
    "glass": (
        "o vidro frontal do automóvel estilhaçou",
        "houve avaria no cristal dianteiro",
        "o parabrisas rachou de ponta a ponta",
        "estilhaçaram o vidro da janela do automóvel",
    ),
    "animal": (
        "colidi com um semovente na rodovia",
        "um bicho se lançou na pista e eu o acertei",
        "choquei contra uma rês que invadiu a estrada",
        "veio uma criação para a pista e não deu para desviar",
    ),
    "vandalism": (
        "houve dano intencional ao automóvel",
        "meu carro foi alvo de atos de depredação",
        "destruíram a pintura do automóvel por maldade",
        "arrombaram e estragaram o interior do automóvel de propósito",
    ),
}

HELD_OUT_GREETINGS = ("saudações", "cordialmente", "prezada seguradora", "senhores")
HELD_OUT_PREAMBLES = (
    "venho pelo presente relatar um evento coberto",
    "solicito a abertura de processo de indenização",
    "informo a ocorrência de um evento danoso",
    "encaminho a comunicação do evento",
)
HELD_OUT_CLOSINGS = (
    "permaneço à disposição",
    "atenciosamente",
    "no aguardo da regulação",
    "solicito retorno quanto à vistoria",
)

#: Every boolean said another way. `vítimas` rather than `feridos`, `rebocado` rather than
#: `guincho`, `ocorrência policial` rather than `boletim de ocorrência` or `BO`.
HELD_OUT_CLAUSES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "third_party_involved": (
        (
            "havia outro condutor envolvido no evento",
            "um segundo automóvel participou do episódio",
            "o condutor do outro automóvel permaneceu no local",
        ),
        (
            "não havia mais ninguém envolvido",
            "nenhum outro automóvel participou",
            "o episódio envolveu apenas o meu automóvel",
        ),
    ),
    "injuries": (
        (
            "houve vítimas no episódio",
            "uma das pessoas precisou de atendimento médico",
            "fui encaminhado ao pronto-socorro",
        ),
        (
            "não houve vítimas",
            "todos saíram ilesos",
            "nenhuma pessoa sofreu qualquer dano físico",
        ),
    ),
    "police_report_filed": (
        (
            "lavrei a ocorrência policial no mesmo dia",
            "a autoridade policial foi comunicada e registrou o fato",
            "o registro policial já foi lavrado",
        ),
        (
            "não lavrei ocorrência policial",
            "a autoridade policial não chegou a ser comunicada",
            "nenhum registro policial foi feito",
        ),
    ),
    "vehicle_drivable": (
        (
            "o automóvel permanece em condições de uso",
            "segui viagem com o próprio automóvel",
            "o veículo continua trafegando sem restrição",
        ),
        (
            "o automóvel teve de ser rebocado",
            "o veículo não reúne condições de trafegar",
            "foi necessário transporte por prancha até a oficina",
        ),
    ),
}

MONTHS_PT = (
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


def _held_out_date(rng: random.Random, received: date) -> tuple[str, date]:
    """Dates said in forms the rule set has no pattern for.

    `há N semanas` and `no dia N do mês passado` are both unambiguous and both absent from the
    rule parser, which handles `há N dias`, a weekday with `passada`, and two numeric forms.
    """
    style = rng.choices(("weeks", "last_month", "ordinal", "numeric"), weights=(7, 7, 5, 3))[0]
    if style == "weeks":
        weeks = rng.randrange(1, 5)
        when = received - timedelta(weeks * 7)
        form = rng.choice((f"há {weeks} semanas", f"faz {weeks} semanas"))
        return (form if weeks > 1 else "há uma semana"), when
    if style == "last_month":
        day = rng.randrange(1, 28)
        year, month = (
            (received.year, received.month - 1)
            if received.month > 1
            else (
                received.year - 1,
                12,
            )
        )
        when = date(year, month, day)
        return f"no dia {day} do mês passado", when
    if style == "ordinal":
        back = rng.randrange(2, 20)
        when = received - timedelta(back)
        return f"em {when.day} de {MONTHS_PT[when.month - 1]} do corrente ano", when
    back = rng.randrange(1, 25)
    when = received - timedelta(back)
    return f"na data de {when.day:02d}.{when.month:02d}.{when.year}", when


def _held_out_amount(rng: random.Random, peril: str) -> tuple[str, Decimal]:
    """Amounts phrased so the rule set's lead-in words do not appear.

    The rule parser reads a spelled amount only after `em`, `por`, `de` or `uns`. `monta a` and
    `importa em` are what a written claim actually says, and neither is in that list.
    """
    low, high = AMOUNT_RANGE[peril]
    value = (
        rng.randrange(low // 100, high // 100 + 1) * 100
        if high <= 10_000
        else rng.randrange(max(low // 1000, 1), high // 1000 + 1) * 1000
    )
    style = rng.choices(("spelled", "written", "formal"), weights=(6, 5, 4))[0]
    if style == "spelled":
        return rng.choice(
            (
                f"o prejuízo monta a {spell_amount(value)} reais",
                f"o dano importa em {spell_amount(value)} reais",
            )
        ), Decimal(value)
    if style == "written":
        return f"a estimativa alcança {spell_amount(value)} reais", Decimal(value)
    text = f"R$ {value:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return rng.choice((f"a monta de {text}", f"o valor apurado: {text}")), Decimal(value)


def _held_out_place(rng: random.Random) -> tuple[str, str, str | None]:
    city, uf = rng.choice(HELD_OUT_CITIES)
    if rng.random() < 0.45:
        return rng.choice((f"no município de {city}, {uf}", f"em {city}/{uf}")), city, uf
    return rng.choice((f"no município de {city}", f"em {city}")), city, None


def _example(rng: random.Random, index: int, first: date, last: date) -> Example:
    received = first + timedelta(rng.randrange((last - first).days + 1))
    peril = rng.choice(tuple(HELD_OUT_OPENINGS))
    clauses: list[str] = []
    withheld: list[str] = []
    occurred_on = city = state = amount = None

    if rng.random() < PRESENCE["occurred_on"]:
        clause, occurred_on = _held_out_date(rng, received)
        clauses.append(clause)
    else:
        withheld.append("occurred_on")

    if rng.random() < PRESENCE["place"]:
        clause, city, state = _held_out_place(rng)
        clauses.append(clause)
    else:
        withheld.append("city")
    if state is None:
        withheld.append("state")

    booleans: dict[str, bool | None] = {}
    for field, applies, positive in (
        ("third_party_involved", None, THIRD_PARTY_POSSIBLE),
        ("injuries", None, INJURIES_POSSIBLE),
        ("police_report_filed", None, None),
        ("vehicle_drivable", DRIVABLE_APPLIES, None),
    ):
        yes, no = HELD_OUT_CLAUSES[field]
        stateable = applies is None or peril in applies
        if stateable and rng.random() < PRESENCE[field]:
            value = rng.random() < 0.55 and (positive is None or peril in positive)
            booleans[field] = value
            clauses.append(rng.choice(yes if value else no))
        else:
            booleans[field] = None
            withheld.append(field)

    if rng.random() < PRESENCE["estimated_amount_brl"]:
        clause, amount = _held_out_amount(rng, peril)
        clauses.append(clause)
    else:
        withheld.append("estimated_amount_brl")

    rng.shuffle(clauses)
    body = ", ".join([rng.choice(HELD_OUT_OPENINGS[peril]), *clauses])

    parts = []
    if rng.random() < 0.55:
        parts.append(rng.choice(HELD_OUT_GREETINGS).capitalize())
    if rng.random() < 0.50:
        parts.append(rng.choice(HELD_OUT_PREAMBLES))
    parts.append(body)
    if rng.random() < 0.45:
        parts.append(rng.choice(HELD_OUT_CLOSINGS))
    text = add_noise(rng, ". ".join(parts) + ".")

    return Example(
        id=f"SHIFT-{index:05d}",
        split="shifted",
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


SHIFTED_SIZE = 200


def generate_shifted(seed: int = 20261006) -> list[Example]:
    """200 records in words no system in this repository has seen.

    A different seed from the main corpus, so the dates and the draws are independent as well as
    the wording.
    """
    from .generate import FIRST_DAY, LAST_DAY

    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[Example] = []
    index = 0
    while len(out) < SHIFTED_SIZE:
        candidate = _example(rng, index, FIRST_DAY, LAST_DAY)
        index += 1
        if candidate.text in seen:
            continue
        seen.add(candidate.text)
        out.append(candidate)
    return out
