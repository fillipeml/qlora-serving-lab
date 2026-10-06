# Working notes

What was measured, in the order it was measured, including the things that did not work. The
README carries the conclusions; this carries the evidence and the dead ends, which is where the
reasoning actually lives.

## The question

Can a small model that fits on one consumer GPU replace a long prompt to a large one, for
structured extraction from short Portuguese documents — and what does it cost either way?

Three systems answer the same 300 records, plus a fourth that uses no model at all. The fourth
is the one most reports leave out and the one that sets the bar.

## The hardware, measured before anything else

A GTX 1660 Ti is TU116: Turing, compute capability 7.5, 6 GiB, and the one Turing chip NVIDIA
shipped **without tensor cores**. That last detail is why everything below happens.

Matrix multiply, 4096², PyTorch 2.11 + CUDA 12.8, **with the card warmed up first** — see the
clock note below, which is the single thing most likely to make these numbers irreproducible:

| dtype | TFLOP/s | |
| --- | --- | --- |
| fp32 | 5.43 | the card's quoted peak is 5.44 |
| bf16 | 2.95 | emulated in software |
| fp16 | 0.63 | the default of nearly every recipe |
| fp16 via fp32 | 4.70 | upcast, multiply, cast back — strictly more work |

fp16 is not slightly slower, it is **nine times slower than fp32** on the same silicon. Casting
fp16 operands up to fp32, multiplying, and casting the result back is **eight times faster than
multiplying them directly**.

Ruled out as explanations:

- Not fp16 in general: element-wise fp16 reaches 247 GB/s against fp32's 251, so the memory path
  is fine and the loss is specific to GEMM.
- Not `allow_fp16_reduced_precision_reduction`: toggling it changes nothing.
- Not cuBLASLt: `DISABLE_ADDMM_CUDA_LT=1` changes nothing.
- Not warmup, not size, not process state: reproduced in isolated processes at 1024, 2048 and
  4096.

### The clock state, which almost cost me the whole measurement

The same sweep, run twice on the same machine an hour apart:

| | fp32 | fp16 | bf16 | fp16 via fp32 | element-wise fp32 |
| --- | --- | --- | --- | --- | --- |
| warm | 5.43 | 0.63 | 2.95 | 4.70 | 251 GB/s |
| cold | 1.31 | 0.15 | 0.69 | 1.22 | 88 GB/s |

Everything is four times lower and **every ratio is identical to two decimal places**
(fp16/fp32 is 0.116 warm and 0.115 cold). An idle NVIDIA card sits in a low power state and
raises its clocks only under sustained load; a benchmark that starts measuring immediately
measures the ramp instead of the card. The per-measurement warmup normally written into these
scripts is sized to pay for kernel selection and takes milliseconds, which is nowhere near
enough.

What makes this worth writing down rather than quietly fixing: the cold numbers are internally
consistent and would have supported every conclusion in this document. The giveaway was that
fp32 came out at 24% of the figure NVIDIA quotes for the card. `bench.py` now spins the clocks
for four seconds before measuring and records the measured fp32 peak as a fraction of the
quoted one, so any run of it says whether it was taken at full clock.

And the guard everybody uses does not see any of it:

```
torch.cuda.is_bf16_supported()                            -> True
torch.cuda.is_bf16_supported(including_emulation=False)   -> False
torch.cuda.get_device_capability()                        -> (7, 5)
```

The default counts software emulation. A recipe that writes
`bf16 = torch.cuda.is_bf16_supported()` therefore takes the bf16 branch on a card with no bf16
units — and on this card that accidentally turns out to be the better of the two, for a reason
its author did not intend.

## What that does to serving

Qwen2.5-0.5B-Instruct, same 8 prompts, batch 1, greedy, median of 8:

| precision | weights | TTFT | decode |
| --- | --- | --- | --- |
| fp32 | 1.84 GiB | 208 ms | 34.6 tok/s |
| bf16 | 0.92 GiB | 273 ms | 27.9 tok/s |
| int8 | 0.59 GiB | 504 ms | 7.8 tok/s |
| fp16 | 0.92 GiB | 1097 ms | 21.4 tok/s |
| nf4 | 0.44 GiB | 1085 ms | 20.4 tok/s |
| nf4-dq | 0.43 GiB | 1101 ms | 18.8 tok/s |

fp32 is fastest on both metrics while using four times the memory of NF4. fp16 — the universal
default — is **5.3× slower to first token** than fp32.

NF4 inherits the problem, because `bnb_4bit_compute_dtype` is conventionally fp16 and the
dequantised weights go straight to the slow kernel. Changing that one line and nothing else:

| nf4 compute dtype | TTFT | decode |
| --- | --- | --- |
| fp16 | 1086 ms | 20.7 tok/s |
| bf16 | 493 ms | 18.9 tok/s |
| fp32 | 473 ms | 20.6 tok/s |

**2.3× faster prefill at identical memory and identical decode rate**, for a one-line change
that every published recipe gets wrong on this hardware. That is why `COMPUTE_DTYPE` in
`infer.py` is fp32 and carries a comment rather than being the usual constant.

## What that does to training

The same question on the training side. Qwen2.5-1.5B-Instruct, NF4 with double quantisation,
LoRA rank 16 on all seven projections, batch 4 with 2 accumulation steps, identical data:

| autocast | seconds per optimiser step | measured over |
| --- | --- | --- |
| off (fp32 compute) | 22.1 | 40 steps |
| bf16 | 140.0 | 3 steps |
| fp16 | not completed | abandoned; see below |

**6.4× slower with bf16 autocast than with none.** The fp16 run was abandoned: the matmul sweep
predicts it is worse still, and at that rate a single optimiser step costs minutes. A recipe
that branches on `torch.cuda.is_bf16_supported()` lands on the 140-second row; one that sets
`fp16=True`, which is what every tutorial for a 6 GiB card says, lands below it.

The three-step bf16 figure carries more warmup per step than the forty-step one, so the true
ratio is somewhat below 6.4×. It is not near 1.

This is why `Settings.autocast` defaults to `"off"`. The 4-bit weights dequantise to fp32 and
the matmul runs there, which on this card is both the fastest and the most numerically stable
option — and it needs no gradient scaler, so no step can be silently skipped to overflow.

## Dead ends and corrections

- **The first protected-token pattern did not protect anything accented.** It was built from the
  words as written — `três`, `sábado` — and matched against text the noise model had already
  stripped of accents, so every accented numeral and weekday could be silently corrupted by a
  typo. Found by `test_protected_tokens_survive_every_noise_setting` at seed 29, which turned
  `três` into `tes`. The corpus was regenerated and the frozen digest updated.
- **Two Wilson expectations in the tests were wrong, not the code.** The numbers written down
  from memory (0.3983, 0.6017 for 50 of 100) are the exact Clopper–Pearson interval, not the
  Wilson one (0.4038, 0.5962). The test now states all three methods' answers so the next person
  does not check one against another by accident.
- **`--max-steps 40` quietly ran 10 steps.** The conversion to epochs divided by the
  accumulation but not by the batch size, so a timing run was a quarter of its stated length.
  Caught because 40 steps finished implausibly fast.
- **Three benchmark lambdas captured loop variables that were deleted afterwards.** Ruff's B023
  and F821 both fired. Harmless as written, wrong the moment the call became lazy.
- **Batching helps or hurts depending on the model, and I nearly reported the wrong one.** On
  Qwen2.5-0.5B, batch 8 is 3.6× faster than batch 1 and the outputs are identical on every
  record checked. On Qwen2.5-1.5B the same comparison came out 4× *slower* — 399 seconds for
  eight records against 105 one at a time. The plausible explanation is that the larger model at
  batch 8 approaches the 6 GiB wall and the allocator starts working for a living, but I have
  one measurement of it and it is not enough to publish a cause, so the README reports the 0.5B
  sweep, which was measured properly across five batch sizes, and says nothing about why the
  1.5B behaves differently.
- **A cap that measured itself.** Generation was capped at 110 new tokens because the target
  record is about 55. The base model answers with a fenced, pretty-printed object that is longer
  than that, so 296 of 300 outputs were cut mid-record and scored as invalid JSON — a number
  that described my cap and not the model. Raised to 256. The interesting part is what it
  uncovered: at 256 tokens only 3 of 8 objects close, and at 512 still exactly 3 of 8. The base
  model is not running out of budget, it is looping, repeating `third_party_involved` and
  `injuries` until it is stopped. That is the real finding, and the bad cap hid it behind a
  number that looked like the same thing.
- **A benchmark that overwrote its own results.** `bench --parts matmul` and
  `bench --parts precision batch` both write `results/bench.json`, and the save replaced the
  file. The second run silently deleted the matmul table from an artifact that still looked
  complete. It merges now.
- **A background run whose output I filtered.** The autocast comparison was piped through a
  `grep` for the final summary line, so a 49-minute bf16 run showed nothing at all and looked
  like a crash. It was not; it was slow. Re-run with nothing filtered, three steps told the
  whole story in seven minutes.
- **The first generator wrote records nobody could have written** — a vandalism claim with
  another vehicle involved, a cracked windscreen that immobilised the car, a stolen car that was
  still drivable. Every one of those teaches a model something false and scores it on something
  nobody would ever write, so the generator now has an explicit table of which fields each peril
  can carry and in which direction.
