# qlora-serving-lab

Fine-tuning and serving a small language model for Portuguese→English claim extraction, on one
six-gigabyte consumer GPU from 2019.

A Brazilian policyholder writes a few lines about what happened to their car. An insurer needs
that as a record: what peril, what date, which municipality, was anyone hurt, was a police report
filed, how much. The message is Portuguese and the record is English, so the job is a translation
as well as a parse.

Five systems do it on the same records, and the report says what each one costs as well as what
it scores. One of them uses no model at all, which is the comparison most reports leave out and
the one that sets the bar. The best one is a composition of two of the others, and which two is
the result this repository exists to produce.

> **Companion repository.** [`claims-intake-rag`](https://github.com/fillipeml/claims-intake-rag)
> takes the other half of the same problem: not turning a short notice into a record, but answering
> questions about one claim from its scanned, OCR'd documents — or refusing to. Same domain, same
> card, the opposite end of the stack: retrieval, hybrid search with a measured ablation, and serving
> over Server-Sent Events, where this one measures training, quantisation and the hardware itself.

## What it found

**A 0.5B model, fine-tuned for nine minutes on a 2019 gaming GPU, beats a hand-written rule
system that knew the data distribution perfectly — and composing the two beats either.**

300 held-out records. Every rate carries its 95% Wilson interval.

| system | JSON valid | **record exact** | field accuracy | **asserted absence** | missed value | ms/record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rules, no model | 100.0% | 62.0% <sub>[56.4, 67.3]</sub> | 95.2% | 1.2% | 6.2% | **0.1** |
| zero-shot 0.5B | 51.3% | 0.0% <sub>[0.0, 1.3]</sub> | 34.3% | **91.4%** | 0.7% | 1,785 |
| few-shot 0.5B | 81.7% | 0.0% <sub>[0.0, 1.3]</sub> | 44.4% | **84.6%** | 2.3% | 957 |
| QLoRA | 99.3% | 77.0% <sub>[71.9, 81.4]</sub> | 97.4% | 0.1% | 0.1% | 627 |
| **QLoRA + deterministic arithmetic** | 99.3% | **97.7%** <sub>[95.3, 98.9]</sub> | **99.8%** | 0.1% | 0.1% | 627 |

The millisecond column is wall clock per record at batch 8, which is how the accuracy runs were
executed; batching does not change a greedy model's answers and was verified not to, record by
record. Single-request latency is a different number and comes from the batch-1 sweep further
down. Note that few-shot is *faster* end to end than zero-shot despite twice the prompt — its
time to first token is worse (3,616 ms against 1,588) and it more than makes that back by
knowing when to stop, which the zero-shot model does not.

Paired exact sign tests, over the records where two systems disagree:

```
rules      -> zero-shot      0 improved, 186 regressed, p < 0.0001
zero-shot  -> few-shot       0 improved,   0 regressed, p = 1.0
few-shot   -> QLoRA        231 improved,   0 regressed, p < 0.0001
QLoRA      -> hybrid        62 improved,   0 regressed, p < 0.0001
rules      -> hybrid       108 improved,   1 regressed, p < 0.0001
```

### Four things worth taking away

**1. Prompting a model this small does not work here, and fails dangerously.** Zero-shot it gets
**no** record completely right, and for 91.4% of the fields the message never mentioned it writes
a value anyway — overwhelmingly `false`. That is not a low score, it is a system that fabricates
facts for an insurance file. Fine-tuning takes it to **0.1%**: one invented field in 1,056.

**2. Four worked examples bought format, and not one correct record.** Few-shot lifts JSON
validity from 51.3% to 81.7% — and against zero-shot it improves **zero** records and regresses
zero (p = 1.0). It also costs 1,175 prompt tokens against 540. If a system has to be shown what
good output looks like on every single call, it has not learned the task; it has learned the
shape.

**3. The two good systems fail in different places, and the places are predictable.** The
adapter beats the rules on every field that requires reading Portuguese, and loses on exactly
one: the date.

| field | rules | QLoRA | hybrid |
| --- | ---: | ---: | ---: |
| peril | 98.0% | 99.7% | 99.7% |
| **occurred_on** | **99.7%** | **79.2%** | **100.0%** |
| city | 99.7% | 100.0% | 100.0% |
| state | 99.7% | 99.7% | 99.7% |
| third_party_involved | 92.0% | 100.0% | 100.0% |
| injuries | 81.7% | 99.7% | 99.7% |
| police_report_filed | 92.7% | 99.7% | 99.7% |
| vehicle_drivable | 93.3% | 99.3% | 99.3% |
| estimated_amount_brl | 100.0% | 99.7% | 100.0% |

*"Sexta passada"* is a subtraction against a known date; it is not language. A 0.5B transformer
does it by pattern and gets it wrong one time in five. Eleven lines of Python do it exactly. So
the hybrid takes the model's record and overrides the two arithmetic fields wherever the
deterministic parser produced a value — and nowhere else, and never where the parser was silent.
That single composition moves 62 records from wrong to right and none the other way.

Two of the model's dates are worth quoting, because they are what the schema rejected rather
than accepted: it produced **`2026-04-31`** and **`2026-02-29`**. April has thirty days and 2026
is not a leap year.

**4. On wording nobody had seen, the rule set collapses and the model does not.** This is the
question the whole repository is built around, because the rules and the corpus share an author
and the adapter trained on the test split's own templates. The `shifted` set is 200 records
saying the same things in different ordinary Portuguese — *assalto* rather than *roubo*,
*rebocado* rather than *guincho*, *vítimas* rather than *feridos*, and twenty-four municipalities
that are not in the gazetteer.

| | field accuracy, familiar | field accuracy, held-out | change | records exact, held-out |
| --- | ---: | ---: | ---: | ---: |
| rules | 95.2% | 46.2% | **−49.0** | **0.0%** |
| QLoRA | 97.4% | 81.6% | **−15.8** | 15.5% |

Field by field, the model wins everywhere on the held-out set, usually by two or three times:

| field | rules | QLoRA |
| --- | ---: | ---: |
| peril | 24.0% | **78.3%** |
| occurred_on | 10.0% | **46.0%** |
| city | 30.5% | **78.3%** |
| state | 69.5% | **99.0%** |
| third_party_involved | 48.5% | **90.4%** |
| injuries | 41.0% | **76.8%** |
| police_report_filed | 45.0% | **93.4%** |
| vehicle_drivable | 68.5% | **76.3%** |
| estimated_amount_brl | 78.5% | **96.5%** |

And the rule set fails in the way a rule set should: on held-out wording its asserted absence
stays at **0.0%** while its missed values go to 76.5%. It has no way to decide that a sentence it
does not recognise means something, so it says nothing. That is safe and useless at the same
time, and it is why the hybrid keeps the model's answer wherever the parser declined — on the
held-out set the composition changes exactly one record, because there is almost nothing for it
to override.

One record, through all five systems:

```
SHIFT-00000   received 2026-09-16

  Saudacoes. sofri um assalto e perdi o automovel, nao havia mais ninguem envolvido,
  lavrei a ocorrencia policial no mesmo dia, na data de 30.08.2026, a estimativa
  alcanca trinta e oito mil reais, no municipio de Imperatriz, MA.

  field                  truth         rules        few-shot       QLoRA        zero-shot
  -------------------------------------------------------------------------------------
  peril                  robbery       x collision  = robbery      = robbery    (no record)
  occurred_on            2026-08-30    - null       x 2026-09-16   = 2026-08-30 (no record)
  city                   Imperatriz    - null       = Imperatriz   = Imperatriz (no record)
  state                  MA            - null       = MA           = MA         (no record)
  third_party_involved   false         - null       = false        = false      (no record)
  injuries               null          = null       + false        = null       (no record)
  police_report_filed    true          - null       x false        = true       (no record)
  vehicle_drivable       null          = null       + false        = null       (no record)
  estimated_amount_brl   38000.0       - null       = 38000.0      = 38000.0    (no record)

  = correct   x wrong   + asserted absence   - missed value
```

`python -m qlora_lab show --split shifted` prints this from the committed results. No GPU.

### A note on the record-level metric

"Record exact" means all nine fields right, which is what a claims handler experiences: one
wrong field and the record is checked by hand, and it does not matter which. It is therefore
brutal, and it is almost exactly what independent per-field errors predict —
0.816⁹ = 16.0% against the 15.5% measured on the held-out set, 0.998⁹ = 98.2% against the hybrid's
97.7%. The model is not failing whole records; it is making scattered, independent field errors.
That is the profile a per-field review queue absorbs, and it is a different operational problem
from a system that gets whole records wrong together.

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

## The hardware, which turned out to be the story

The card is a **GTX 1660 Ti**: Turing, compute capability 7.5, 6 GiB — and TU116, the one Turing
chip NVIDIA shipped **without tensor cores**. Everything below follows from that.

Square matrix multiply, PyTorch 2.11 + CUDA 12.8, measured with the clocks warmed up (fp32 came
out at 94% of the 5.44 TFLOP/s NVIDIA quotes for the card, which is how the benchmark says it was
not throttled):

| dtype | 1024² | 2048² | 4096² |
| --- | ---: | ---: | ---: |
| fp32 | 3.13 | 4.79 | **5.13** |
| bf16 — emulated in software | 2.74 | 2.81 | 2.83 |
| **fp16 — the default of nearly every recipe** | 0.56 | 0.56 | **0.62** |
| fp16 upcast to fp32, multiplied, cast back | 2.83 | 4.54 | **4.95** |

fp16 is not slightly slower. It is **eight times slower than fp32** on the same silicon, and
casting fp16 operands *up* to fp32, multiplying, and casting the result back — strictly more
work — is **eight times faster than multiplying them directly**.

Three explanations ruled out:

- **Not fp16 in general.** Element-wise fp16 reaches 245.4 GB/s against fp32's 249.1. The memory
  path is fine; the loss is specific to GEMM.
- **Not reduced-precision accumulation.** Toggling
  `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` changes nothing.
- **Not cuBLASLt.** `DISABLE_ADDMM_CUDA_LT=1` changes nothing.

And the guard everyone branches on does not see any of it:

```python
torch.cuda.is_bf16_supported()  # True
torch.cuda.is_bf16_supported(including_emulation=False)  # False
torch.cuda.get_device_capability()  # (7, 5)
```

The default counts software emulation. A recipe that writes
`bf16 = torch.cuda.is_bf16_supported()` takes the bf16 branch on a card with no bf16 units — and
on this card that is accidentally the better of the two, for a reason its author did not intend.

### What it costs at serving time

Qwen2.5-0.5B-Instruct, the same sixteen prompts, batch 1, greedy:

| precision | weights | peak | time to first token | decode |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 1.84 GiB | 2.18 GiB | 210.9 ms | 43.9 tok/s |
| **nf4** | **0.69 GiB** | **1.03 GiB** | **215.4 ms** | **35.5 tok/s** |
| nf4 + double quant | 0.68 GiB | 1.02 GiB | 217.5 ms | 35.2 tok/s |
| bf16 | 0.92 GiB | 1.09 GiB | 239.1 ms | 40.1 tok/s |
| int8 | 0.84 GiB | 1.18 GiB | 259.5 ms | **8.5 tok/s** |
| **fp16** | 0.92 GiB | 1.09 GiB | **1083.5 ms** | 34.2 tok/s |

fp16 — the universal default — is **5.1× slower to first token** than fp32 at the same memory as
bf16. NF4 lands within 2% of fp32's latency at **37% of the weight memory**, which is what makes
it the configuration this repository serves on. `int8` is the other outlier: its time to first
token is fine and its decode is four times slower than everything else, which is the known cost
of the mixed-precision decomposition `LLM.int8()` does per matmul.

### The one line every 4-bit recipe sets without thinking

`bnb_4bit_compute_dtype` decides what the 4-bit weights are dequantised *into* before they are
multiplied. Every published QLoRA recipe sets it to fp16 or bf16 and moves on. The same adapter,
the same weights, the same prompts — only that line changes:

| `bnb_4bit_compute_dtype` | weights | time to first token | decode |
| --- | ---: | ---: | ---: |
| **fp16** — the convention | **0.44 GiB** | **1083.3 ms** | 26.7 tok/s |
| bf16 | 0.44 GiB | 246.9 ms | 31.8 tok/s |
| fp32 | 0.69 GiB | **219.2 ms** | **34.2 tok/s** |

The weights stay four bits either way. **bf16 is 4.4× faster to first token than fp16 at exactly
the same memory**, because the dequantised values go straight to the GEMM kernel measured above.
fp32 is faster still and costs 0.25 GiB more, because the layers that are never quantised — the
embeddings, the head, the norms — are held in the compute dtype.

So on this card the recommendation is not "use fp32", it is **anything but fp16**: bf16 when
memory binds, fp32 when it does not. This repository defaults to fp32 because that is what every
number above was measured with, and the comment in `infer.py` says so rather than presenting a
convention as a decision.

### The same question, in training

Qwen2.5-0.5B, NF4 with double quantisation, LoRA rank 16 on all seven projections, batch 4,
five optimiser steps each:

| autocast | per step | peak | reached |
| --- | ---: | ---: | ---: |
| **off** — fp32 compute | **2.07 s** | 2.19 GiB | 0.8537 |
| bf16 | 2.66 s | 2.44 GiB | 0.8579 |
| **fp16** — what every tutorial sets | **9.17 s** | 2.44 GiB | 0.8537 |

**fp16 autocast is 4.4× slower than none and reaches exactly the same loss.** It is pure cost.
No step was skipped to gradient overflow in any configuration, which is the other thing fp16 can
take from you silently and did not here.

A recipe that writes `bf16 = torch.cuda.is_bf16_supported()` lands on the middle row — 1.29×
slower than doing nothing, for a feature the card does not have. One that writes `fp16=True`,
which is what every tutorial for a 6 GiB card says, lands on the bottom.

### Batch size: throughput against latency, with a hard wall

NF4, the same prompts, 100 new tokens:

| batch | records/s | time to first token | peak VRAM |
| ---: | ---: | ---: | ---: |
| 1 | 0.31 | 217 ms | 1.03 GiB |
| 2 | 0.59 | 406 ms | 1.36 GiB |
| 4 | 0.89 | 606 ms | 2.02 GiB |
| 8 | 1.04 | 1,641 ms | 3.34 GiB |
| 16 | **1.48** | **3,060 ms** | **5.96 GiB** |

Batching buys 4.8× the throughput and costs 14× the latency, and at 16 the card is 60 MiB from
its limit. There is no 32. That is what batching is worth **without** a serving engine, which is
the comparison the next section makes.

---

### What a serving engine buys, measured

Everything above is HuggingFace `transformers` with a fixed batch: every request in a batch waits
for the batch to be assembled and then for its slowest member to finish. A serving engine does
not work that way — vLLM admits and retires requests continuously, so a request that arrives late
does not wait for one that arrived early, and a request that finishes early frees its slot at
once. The claim is not that it is faster in aggregate. It is that **you stop paying latency for
throughput**.

vLLM 0.31.0, same card, same model, same fp32, same 64 Portuguese claim prompts, same 100 new
tokens, greedy. Run under WSL2 because vLLM is Linux-only; the card is passed through, and
`nvidia-smi` inside Ubuntu reports the same GTX 1660 Ti.

| | fixed batch (transformers) | | continuous batching (vLLM) | |
| ---: | ---: | ---: | ---: | ---: |
| **concurrency** | **records/s** | **TTFT** | **records/s** | **TTFT p50** |
| 1 | 0.36 | 206 ms | **0.80** | **77 ms** |
| 2 | 0.64 | 392 ms | **1.63** | **59 ms** |
| 4 | 1.14 | 590 ms | **3.29** | **62 ms** |
| 8 | 1.16 | 1,947 ms | **6.18** | **75 ms** |
| 16 | 1.53 | 3,492 ms | **7.08** | **116 ms** |

At sixteen concurrent requests vLLM delivers **4.6× the throughput at one thirtieth of the median
time to first token**. The shape matters more than either number: the fixed batch's latency grows
17× from one request to sixteen, because that is what waiting for a batch means, while vLLM's
median stays between 59 and 116 ms throughout. That is the whole argument for a serving engine,
and it is visible here on a card that cost less than a monitor.

The memory story is the same shape. The fixed batch peaked at 7.11 GiB on a 6,144 MiB card at
batch 16 — the Windows display driver pages to system memory rather than failing, which is why
throughput barely moves from batch 8 to 16 while latency nearly doubles again. vLLM reports
`GPU KV cache size: 111,616 tokens, Maximum concurrency for 2,048 tokens per request: 54.50x`.
Paged attention is why: it allocates the cache in blocks rather than reserving a rectangle big
enough for every sequence's worst case.

### And the engine's own default is the slow path here

vLLM resolves `--dtype auto` against the model's declared dtype. Qwen2.5 asks for bfloat16, the
card has none, and it falls back to **float16** — which on this chip is the kernel the top of this
report measures at an eighth of fp32. Same engine, same everything, one flag:

| concurrency | fp32 records/s | fp16 records/s | fp32 TTFT p50 | fp16 TTFT p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.80 | 0.87 | 77 ms | 115 ms |
| 2 | 1.63 | **0.29** | 59 ms | 246 ms |
| 4 | 3.29 | 0.56 | 62 ms | 341 ms |
| 8 | 6.18 | 1.09 | 75 ms | 345 ms |
| 16 | **7.08** | **2.09** | 116 ms | 352 ms |

**3.4× the throughput for one flag**, and the shape of the fp16 column is the more interesting
half. From concurrency 2 onward it doubles cleanly — 0.29, 0.56, 1.09, 2.09 — so it scales
perfectly. The outlier is concurrency **1**, which is three times faster per request than every
level above it: one request alone takes 1.16 s, two together take 6.9 s each.

That is what a card without tensor cores looks like from above. With a single sequence, decode is
a matrix-*vector* product. From two sequences on, it is a matrix-*matrix* product — and that is
the fp16 GEMM measured at 0.62 TFLOP/s against fp32's 5.13 at the top of this report. **The fp16
penalty on this chip does not appear until you batch**, which is exactly why a single-request
benchmark would never find it, and why the same flag costs nothing at concurrency 1 and
everything at concurrency 16.

In fp32 the same transition is free: 0.80 to 1.63 is a clean doubling, with no cliff anywhere.

fp16 does buy something real — `GPU KV cache size: 311,264 tokens, Maximum concurrency 151.98x`
against fp32's 111,616 and 54.50x, because a half-precision cache is half the bytes. On this card
that is 2.8× the cache for 3.4× less throughput. On a card with tensor cores it would be free.

### What the log says, and why it is quoted rather than summarised

```
Upcasting torch.bfloat16 to torch.float32.
Turing devices tensor cores do not support float32 matmul. To workaround this limitation,
  vLLM will set 'ieee' input precision for chunked prefill triton kernels.
Cannot use FA version 2 is not supported due to FA2 is only supported on devices with
  compute capability >= 8
Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION'].
GPU KV cache size: 111,616 tokens, Maximum concurrency for 2,048 tokens per request: 54.50x
```

vLLM knows about this class of card and says so: there is a branch in it written for
`get_device_capability() == (7, 5)`. FlashAttention is refused outright rather than silently
degraded, and the Triton backend is what remains. None of that is a problem to be solved; it is
the configuration the numbers above were measured in, and a benchmark that does not record it
cannot be compared with anything.

### Reproducing it

Under WSL2 on Windows, with the card passed through:

```bash
# A C compiler, because Triton compiles its kernels at run time:
sudo apt install -y build-essential

uv venv --python 3.12 ~/.venvs/vllm
uv pip install --python ~/.venvs/vllm/bin/python vllm aiohttp --torch-backend=auto
python scripts/vllm_bench.py --dtype float32 --requests 64 --levels 1 2 4 8 16 --enforce-eager
python scripts/vllm_bench.py --dtype float16 --requests 64 --levels 1 2 4 8 16 --enforce-eager
```

Two things that will stop you, both of which did:

- **`gpu_memory_utilization` is a fraction of total VRAM, not of free.** On a 6 GiB card with a
  desktop holding about a gigabyte, 0.85 asks for 5.1 GiB of the 5.01 GiB that exist and the
  engine refuses to start — correctly, and with a message that says exactly that. 0.78 fits.
- **`--enforce-eager` is used here and is not free.** It disables `torch.compile` and CUDA-graph
  capture, which cost minutes of first-start and extra VRAM this card does not have. It also
  removes part of what a production deployment would be measuring, so these numbers are a floor
  on what vLLM can do here rather than its best.

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

- **TGI and Triton.** vLLM is measured above, under WSL2. Hugging Face TGI and NVIDIA Triton are
  not, and no claim is made about them. Nor is TensorRT-LLM, which requires compute capability
  8.0 and so cannot run on this card at all.
- **vLLM at its best.** The numbers above use `--enforce-eager`, which disables `torch.compile`
  and CUDA-graph capture to fit a 6 GiB card and keep first-start to minutes. They are a floor on
  what vLLM does here, not its ceiling.
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
