"""Loading a model at six precisions, and timing it honestly.

**Time to first token and tokens per second are two different machines.** Prefill runs the whole
prompt through the network at once — one large matrix multiply per layer, compute-bound. Decode
runs one token at a time — a tall thin multiply per layer, and what limits it is how fast the
weights can be read from memory, not how fast they can be multiplied. A quantisation that halves
the bytes speeds decode up and can leave prefill alone or make it worse. So both are measured,
and the report never prints one without the other.

**On this card that distinction is not academic.** The GTX 1660 Ti is TU116, the one Turing chip
NVIDIA shipped without tensor cores, and its fp16 GEMM path in CUDA 12.8 runs at roughly a ninth
of the fp32 rate — see `bench.py` and the README. Every recipe on the internet sets
`torch_dtype=float16` on a card like this. The measurement in this repository is what that
costs, split by phase, because the answer is not the same for prefill and decode.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import torch

#: Every precision the lab compares. `fp16` is the default of nearly every recipe; `bf16` on
#: this card is emulated in software, which `torch.cuda.is_bf16_supported()` does not say.
PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16", "int8", "nf4", "nf4-dq")

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def bf16_is_real() -> bool:
    """Whether the GPU has bf16 in hardware, rather than emulated in software.

    `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and so answers True
    on a card with no bf16 units at all. Recipes branch on it to choose a training dtype, which
    means the branch is not measuring what its author thinks. The keyword argument that tells
    the truth is recent; older versions have no way to ask, hence the fallback on compute
    capability, where 8.0 (Ampere) is the first with bf16 in silicon.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return torch.cuda.get_device_capability(0)[0] >= 8


@dataclass
class Loaded:
    model: Any
    tokenizer: Any
    precision: str
    model_id: str
    #: Bytes of VRAM the weights occupy, measured rather than computed from the parameter count:
    #: a quantised layer keeps its scales and zero points too, and a formula forgets them.
    weight_bytes: int
    load_seconds: float

    @property
    def weight_gib(self) -> float:
        return self.weight_bytes / 2**30


#: What a 4-bit layer dequantises into before it multiplies. Every published QLoRA recipe sets
#: this to fp16 or bf16 and moves on; on this card it is the single largest serving decision
#: available, because the 4-bit weights are unpacked and then handed to exactly the GEMM kernel
#: that `bench.py` shows to be nine times slower in fp16 than in fp32. Changing this one line
#: and nothing else takes time to first token from 1083 ms to 247 ms at *identical* memory
#: (bf16) or to 219 ms for 0.25 GiB more (fp32, which also holds the unquantised embeddings and
#: head). The recommendation on this card is not "use fp32", it is "anything but fp16"; bf16
#: when memory binds and fp32 when it does not.
#:
#: fp32 here because it is what every number in the README was measured with, and a default
#: that disagrees with the published measurements is worse than a slower one.
COMPUTE_DTYPE = torch.float32


def _quantisation_config(precision: str, compute_dtype: torch.dtype | None = None):
    from transformers import BitsAndBytesConfig

    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision in {"nf4", "nf4-dq"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # Double quantisation quantises the quantisation constants themselves. It saves
            # about half a bit per parameter, which on a 1.5B model is tens of megabytes — small
            # in absolute terms and decisive when the budget is 6 GiB and an optimiser state has
            # to fit beside the weights.
            bnb_4bit_use_double_quant=precision == "nf4-dq",
            bnb_4bit_compute_dtype=compute_dtype or COMPUTE_DTYPE,
        )
    return None


def load(
    model_id: str,
    precision: str = "nf4",
    adapter: str | None = None,
    compute_dtype: torch.dtype | None = None,
) -> Loaded:
    """Load a model at one precision, optionally with a LoRA adapter on top."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding, because a batch is generated from the right-hand end: with right padding the
    # model continues from the pad tokens and every sequence but the longest produces rubbish.
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {"device_map": {"": 0}}
    config = _quantisation_config(precision, compute_dtype)
    if config is not None:
        kwargs["quantization_config"] = config
        kwargs["dtype"] = compute_dtype or COMPUTE_DTYPE
    else:
        kwargs["dtype"] = DTYPES[precision]

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    torch.cuda.synchronize()

    return Loaded(
        model=model,
        tokenizer=tokenizer,
        precision=precision,
        model_id=model_id,
        weight_bytes=torch.cuda.memory_allocated() - before,
        load_seconds=time.perf_counter() - started,
    )


@dataclass
class Timing:
    """What one generation cost."""

    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    total_ms: float
    #: Peak VRAM during the call, including the key-value cache, which is the thing that
    #: actually runs a 6 GiB card out of memory at batch sizes nobody expects.
    peak_bytes: int = 0

    @property
    def decode_tokens_per_second(self) -> float:
        """Tokens per second after the first one.

        Measured from the first token rather than from the request, because including prefill
        averages two different rates into a number that describes neither.
        """
        decode_ms = self.total_ms - self.ttft_ms
        if decode_ms <= 0 or self.generated_tokens <= 1:
            return 0.0
        return (self.generated_tokens - 1) * 1000 / decode_ms


@dataclass
class Run:
    outputs: list[str] = field(default_factory=list)
    timings: list[Timing] = field(default_factory=list)

    def median_ttft(self) -> float:
        values = sorted(t.ttft_ms for t in self.timings)
        return values[len(values) // 2] if values else 0.0

    def median_decode_rate(self) -> float:
        values = sorted(t.decode_tokens_per_second for t in self.timings)
        return values[len(values) // 2] if values else 0.0


def _trim(text: str) -> str:
    """Cut the completion at the end of the first JSON object.

    A model with no stop token keeps going — another object, a word of commentary, the beginning
    of a new message. Counting that as a parse failure would measure the absence of a stop
    sequence rather than the quality of the extraction, so the brace depth is tracked and the
    text ends where the object does.
    """
    depth = 0
    for index, character in enumerate(text):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return text


@torch.inference_mode()
def generate(
    loaded: Loaded,
    prompts: list[str],
    max_new_tokens: int = 160,
    batch_size: int = 1,
    chat: bool = False,
) -> Run:
    """Generate for every prompt, timing each batch.

    Greedy, always. Sampling would make the accuracy table a function of the seed, and a report
    that moves when nothing changed cannot show that something did.
    """
    tokenizer = loaded.tokenizer
    run = Run()
    torch.cuda.reset_peak_memory_stats()

    rendered = prompts
    if chat:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]

    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True).to("cuda")
        prompt_length = encoded["input_ids"].shape[1]

        torch.cuda.synchronize()
        began = time.perf_counter()

        # Prefill alone, so time to first token is measured rather than inferred. `generate`
        # with one new token would re-enter the whole stack; a direct forward pass is the
        # prefill and nothing else.
        prefill = loaded.model(**encoded, use_cache=True)
        first = prefill.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - began) * 1000
        del prefill, first

        # Timed from here, not from `began`. `generate` runs its own prefill, so measuring the
        # whole block would charge the prompt twice and the decode window — the difference
        # between the two — would be inflated by exactly one prefill. Harmless at a 100-token
        # prompt and a 44% error on a 1,190-token one, which is the case the report cares about.
        generation_started = time.perf_counter()
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - generation_started) * 1000

        new_tokens = generated[:, prompt_length:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        produced = int((new_tokens != tokenizer.pad_token_id).sum(dim=1).float().mean().item())

        for text in texts:
            run.outputs.append(_trim(text.strip()))
        for _ in texts:
            run.timings.append(
                Timing(
                    prompt_tokens=prompt_length,
                    generated_tokens=max(produced, 1),
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    peak_bytes=torch.cuda.max_memory_allocated(),
                )
            )

    return run


def release(loaded: Loaded | None) -> None:
    """Free the card between configurations.

    Without this the second model in a sweep loads beside the first and a 6 GiB card reports an
    out-of-memory error that has nothing to do with the configuration being measured.
    """
    if loaded is not None:
        del loaded.model
        del loaded.tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
