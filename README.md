# qlora-serving-lab

Fine-tuning and serving a small language model for Portuguese→English claim extraction, on one
six-gigabyte consumer GPU from 2019.

A Brazilian policyholder writes a few lines about what happened to their car. An insurer needs
that as a record: what peril, what date, which municipality, was anyone hurt, was a police report
filed, how much. The message is Portuguese and the record is English, so the job is a translation
as well as a parse.

Four systems do it on the same records, and the report says what each one costs as well as what
it scores. One of them uses no model at all, which is the comparison most reports leave out and
the one that sets the bar.

<!-- RESULTS -->

---

## What it measures, and why each number is split in two

**Accuracy is four outcomes, not one.** For every field, exactly one of these happened:

| | |
| --- | --- |
| **correct** | the answer matches, including both being null |
| **wrong** | the text said something and the system said something else |
| **asserted absence** | the text said nothing and the system filled the field in anyway |
| **missed value** | the text said something and the system left it null |

A single accuracy figure treats the last three as the same failure. They are not. *"Não houve
feridos"* is `injuries: false` — somebody looked and said so. A message that never mentions
injuries is `injuries: null` — nobody looked. Writing `false` in the second case puts a fact on
an insurance file that no human ever stated, and nothing downstream can tell the two apart. That
is the error this repository is built to make visible, because it is the one a small model makes
most and the one a single accuracy number is least able to show.

**Latency is two numbers, not one.** Prefill runs the whole prompt through the network at once
and is limited by arithmetic; decode runs one token at a time and is limited by how fast the
weights can be read from memory. A quantisation that halves the bytes speeds one up and can leave
the other alone or make it worse. Time to first token and tokens per second are therefore always
printed together.

**Every rate carries a Wilson interval,** and every comparison between two systems is a paired
exact sign test over the records where they disagree — the systems answered the same records, and
treating them as independent samples throws the pairing away.

---

## The corpus

A real notice of loss is somebody's accident: a date, a place, a vehicle, often an injury. There
is no version of that corpus that can be committed to a public repository. So it is generated —
and generating it is not a compromise here, it is the better instrument.

The record is drawn **first** and the message is written **from** it. The label is therefore
exact by construction rather than by annotation, and — the part that matters — the generator
knows which fields it deliberately withheld. Without that there is no denominator for asserted
absence, and the error that motivates the whole report cannot be counted at all.

The messages read like messages. Accents are dropped, because a phone keyboard drops them. There
are typos. Dates arrive as *"ontem"*, *"sexta passada"*, *"há três dias"* and *"no dia 12 de
março"*, and the label is the resolved ISO date. Amounts arrive as `R$ 3.500,00`, as *"uns 3
mil"*, and as *"três mil e quinhentos"*. All of that noise is applied **after** the record
exists, and never to a token that carries meaning: a typo that ate the *não* in *"não houve
feridos"* would silently change the truth, so negations, numerals, weekday names and place names
are protected, and a test asserts it over four hundred seeds.

The generator also refuses to write a record a person could not have written. A stolen car is
never also drivable. A fire never has another vehicle involved. A chipped windscreen and a total
loss are three orders of magnitude apart in value. Each of those constraints exists because the
first version violated it, and every such record would teach a model something false while
scoring it on something nobody would ever write.

One field is the bilingual test. Portuguese separates **roubo** (taken by force or threat) from
**furto** (taken without confrontation); English says *theft* for both. An insurer does not
collapse them — different reserve, different fraud profile, different police process. A model
trained mostly on English has no reason to split them, and whether it does is the most direct
available evidence that it is reading Portuguese rather than pattern-matching it.

**Four splits**, written by `python -m qlora_lab data`, all deterministic, all pinned by a digest
in CI so a change to the generator cannot silently move every number in this report:

| split | records | what it is |
| --- | --- | --- |
| `train` | 1,000 | what the adapter is trained on |
| `validation` | 200 | training loss, and one sanity check |
| `test` | 300 | the main table; touched once, at the end |
| `shifted` | 200 | **the same task in words no system has seen** |

---

## The four systems

| | what it is | prompt |
| --- | --- | --- |
| **rules** | Regular expressions, a municipality gazetteer and a date parser. No model, about a millisecond a record. | — |
| **zero-shot** | Qwen2.5-0.5B-Instruct, given an instruction that spells out the schema, the eight perils, the date arithmetic and the null rule. | 540 tokens |
| **few-shot** | The same, plus four worked examples from the training split, one per peril. | 1,175 tokens |
| **tuned** | The same base model with a LoRA adapter — 8.8M trainable parameters against a 494M-parameter base, 1.8%. | 88 tokens |

**Read the rule baseline as a ceiling, not a baseline.** The same person wrote it and wrote the
generator. The cues it looks for are the cues the generator emits, and no production rule set is
ever in that position. It measures the best rules could possibly do against a distribution they
already know perfectly. A model that merely matches it has not earned its GPU.

That cuts both ways, and it is why the `shifted` split exists. The adapter has the mirror-image
advantage: it trained on a thousand messages drawn from the same templates the test split comes
from. Both numbers are optimistic, in ways that do not cancel.

---

<!-- HARDWARE -->

---

## Running it

The offline half — corpus, rule baseline, metrics, statistics and the record inspector — has
**no dependencies at all** and needs no GPU:

```bash
pip install -e ".[dev]"
python -m qlora_lab data                      # write the four splits
python -m qlora_lab baseline --split test     # the rule baseline
python -m qlora_lab show --split test         # one record through every saved system
pytest -q
```

The GPU half needs the extra:

```bash
pip install -e ".[gpu]"
python -m qlora_lab bench                     # matmul, six precisions, batch sizes
python -m qlora_lab train --epochs 2          # QLoRA
python -m qlora_lab tuned --split test --adapter adapters/qwen2.5-0.5b-r16
```

`scripts/run_all.sh` runs every step in the order that produced the numbers above, and
[`docs/NOTES.md`](docs/NOTES.md) is the working record: what was measured, in what order, and
everything that went wrong on the way — including two mistakes that produced confident,
internally consistent, wrong artefacts.

Every command defaults to the `validation` split. A test number should be produced on purpose,
once, and the way to make that true is to make it the inconvenient option.

---

## What is not here

- **vLLM, TGI and Triton.** vLLM requires compute capability 7.5, which this card meets exactly,
  and Linux, which this machine does not run. Continuous batching and paged attention are
  therefore not measured and no claim is made about them. The batch sweep is what batching buys
  without a serving engine.
- **A larger model, trained.** Qwen2.5-1.5B fits, and supplied the training-mode timings in
  `docs/NOTES.md`, but a full run on it is about an hour and a half on this card. A scaling row
  measured once is not worth that in a report about what a cheap card can do.
- **A frontier-model ceiling.** Sending the test split to a hosted API would give an upper bound
  and a cost comparison. It would also spend money on somebody else's infrastructure to answer a
  question this repository is not asking, which is what a 0.5B model can do on hardware you
  already own.
- **Any claim about production traffic.** The corpus is generated. It is faithful to how these
  messages are written; it is not a sample of them.

---

## Licence

MIT. The corpus is generated and contains no real person, vehicle, policy or claim.
