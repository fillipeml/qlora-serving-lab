"""What the hardware actually does, as opposed to what the recipe assumes.

Two benchmarks. The first is a matrix multiply at three dtypes, which is the smallest thing that
can show the problem. The second is the same model loaded six ways and asked the same questions,
which is the only thing that can show whether the problem matters.

The finding that motivated this module, on a GTX 1660 Ti (TU116, compute 7.5, the one Turing
chip NVIDIA shipped without tensor cores):

    fp32    5.43 TFLOP/s      the card's quoted peak is 5.44
    bf16    2.95 TFLOP/s      emulated in software, and still five times fp16
    fp16    0.58 TFLOP/s      the default of nearly every recipe

fp16 is not slightly slower, it is nine times slower than fp32 on the same silicon, and
element-wise fp16 runs at full memory bandwidth — so it is the GEMM kernel, not fp16 as such.
Casting fp16 operands up to fp32, multiplying, and casting the result back is eight times faster
than multiplying them directly. `matmul_sweep` reproduces all of it in about twenty seconds.

None of this is visible through `torch.cuda.is_bf16_supported()`, which answers True here
because its default counts emulation.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class MatmulResult:
    dtype: str
    size: int
    tflops: float
    note: str = ""


def _time(call, iterations: int) -> float:
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def matmul_sweep(sizes: tuple[int, ...] = (1024, 2048, 4096)) -> list[MatmulResult]:
    """A square matrix multiply at each dtype, plus the upcast path, at each size.

    Warmed up before every measurement, because the first call at a new shape and dtype pays for
    kernel selection and would otherwise be charged to the dtype.
    """
    results: list[MatmulResult] = []
    for size in sizes:
        iterations = 100 if size <= 2048 else 30
        for name, dtype in (
            ("fp32", torch.float32),
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
        ):
            a = torch.randn(size, size, device="cuda", dtype=dtype)
            b = torch.randn(size, size, device="cuda", dtype=dtype)
            elapsed = _time(lambda a=a, b=b: a @ b, iterations)
            results.append(MatmulResult(name, size, 2 * size**3 * iterations / elapsed / 1e12))
            del a, b
            torch.cuda.empty_cache()

        # The same fp16 operands, multiplied in fp32 and cast back. Strictly more work, and on
        # this card strictly faster — which is the clearest statement of where the problem is.
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        elapsed = _time(lambda a=a, b=b: (a.float() @ b.float()).half(), iterations)
        results.append(
            MatmulResult(
                "fp16 via fp32",
                size,
                2 * size**3 * iterations / elapsed / 1e12,
                "upcast, multiply, downcast",
            )
        )
        del a, b
        torch.cuda.empty_cache()
    return results


def bandwidth_check(size: int = 2048) -> list[MatmulResult]:
    """Element-wise multiply at two dtypes, reported as GB/s in the `tflops` field.

    Here to answer the obvious objection to the matmul sweep: perhaps fp16 is simply broken on
    this card. It is not — element-wise fp16 reaches the same bandwidth as fp32, so the loss is
    specific to the GEMM path.
    """
    results = []
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        a = torch.randn(size * size, device="cuda", dtype=dtype)
        b = torch.randn_like(a)
        elapsed = _time(lambda a=a, b=b: a * b, 100)
        moved = 3 * a.numel() * a.element_size() * 100
        results.append(MatmulResult(name, size, moved / elapsed / 1e9, "GB/s, element-wise"))
        del a, b
        torch.cuda.empty_cache()
    return results


@dataclass
class ServingResult:
    precision: str
    weight_gib: float
    peak_gib: float
    ttft_ms: float
    decode_tokens_per_second: float
    load_seconds: float


def serving_sweep(
    model_id: str,
    prompts_: list[str],
    precisions: tuple[str, ...] = ("fp32", "fp16", "bf16", "int8", "nf4", "nf4-dq"),
    chat: bool = True,
    max_new_tokens: int = 100,
) -> list[ServingResult]:
    """The same model and the same prompts at every precision.

    Each precision is loaded, measured and released before the next is loaded. Keeping two on
    the card at once turns the last few configurations into out-of-memory errors that have
    nothing to do with the configuration being measured.
    """
    from . import infer

    results: list[ServingResult] = []
    for precision in precisions:
        loaded = None
        try:
            loaded = infer.load(model_id, precision)
            run = infer.generate(
                loaded, prompts_, max_new_tokens=max_new_tokens, batch_size=1, chat=chat
            )
            results.append(
                ServingResult(
                    precision=precision,
                    weight_gib=loaded.weight_gib,
                    peak_gib=max(t.peak_bytes for t in run.timings) / 2**30,
                    ttft_ms=run.median_ttft(),
                    decode_tokens_per_second=run.median_decode_rate(),
                    load_seconds=loaded.load_seconds,
                )
            )
        finally:
            infer.release(loaded)
            gc.collect()
    return results


def batch_sweep(
    model_id: str,
    prompts_: list[str],
    precision: str = "nf4",
    sizes: tuple[int, ...] = (1, 2, 4, 8, 16),
    chat: bool = True,
    max_new_tokens: int = 100,
) -> list[dict]:
    """Where a 6 GiB card runs out, and what batching buys before it does.

    Batching is the one throughput lever available without a serving engine: the weights are
    read once per batch instead of once per request, so decode gets most of its cost back. What
    it does not improve is latency for the single request — which is why both are reported, and
    why an out-of-memory error is recorded as a row rather than raised.
    """
    from . import infer

    rows: list[dict] = []
    for size in sizes:
        loaded = None
        try:
            loaded = infer.load(model_id, precision)
            started = time.perf_counter()
            run = infer.generate(
                loaded, prompts_, max_new_tokens=max_new_tokens, batch_size=size, chat=chat
            )
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "batch_size": size,
                    "records": len(prompts_),
                    "seconds": round(elapsed, 2),
                    "records_per_second": round(len(prompts_) / elapsed, 2),
                    "ttft_ms": round(run.median_ttft(), 1),
                    "peak_gib": round(max(t.peak_bytes for t in run.timings) / 2**30, 2),
                }
            )
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": size, "records": len(prompts_), "error": "out of memory"})
        finally:
            infer.release(loaded)
            gc.collect()
    return rows


def describe_device() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    from .infer import bf16_is_real

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "cuda": True,
        "device": name,
        "compute_capability": f"{major}.{minor}",
        "total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "torch": torch.__version__,
        "is_bf16_supported_default": torch.cuda.is_bf16_supported(),
        "bf16_in_hardware": bf16_is_real(),
    }


def save(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def as_json(results: list) -> list[dict]:
    return [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in results]
