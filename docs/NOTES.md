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

Matrix multiply, 4096², PyTorch 2.11 + CUDA 12.8:

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
- **The first generator wrote records nobody could have written** — a vandalism claim with
  another vehicle involved, a cracked windscreen that immobilised the car, a stolen car that was
  still drivable. Every one of those teaches a model something false and scores it on something
  nobody would ever write, so the generator now has an explicit table of which fields each peril
  can carry and in which direction.
