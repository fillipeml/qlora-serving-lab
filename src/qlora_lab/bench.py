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


def warm_up_clocks(seconds: float = 4.0) -> None:
    """Spin the GPU up before measuring anything.

    An idle NVIDIA card sits in a low power state and raises its clocks only once there is
    sustained work. A benchmark that starts measuring immediately measures the ramp. This
    repository has both versions on record: run cold, the sweep reported 1.31 TFLOP/s fp32;
    run warm, 5.43 against a quoted peak of 5.44 — and every ratio between dtypes was identical
    to two decimal places, which is exactly the shape that gets mistaken for a real difference.

    Four seconds of large fp32 multiplies is enough on this card. The per-measurement warmup in
    `_time` is not: it is sized to pay for kernel selection, which takes milliseconds.
    """
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()


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
    warm_up_clocks()
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
    warm_up_clocks()
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

    warm_up_clocks()
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


#: What NVIDIA quotes for this card at boost clock, in TFLOP/s fp32: 1536 shaders x 2 flops x
#: 1.77 GHz. Used only to say whether a particular run was measured at full clock, because a
#: throttled run gives numbers that are uniformly low and ratios that are unchanged — which is
#: exactly the shape that gets mistaken for a real difference between two configurations.
QUOTED_FP32_TFLOPS = 5.44


def clock_health(results: list[MatmulResult]) -> dict:
    """How close the largest fp32 multiply came to the card's quoted peak.

    One run of this repository's own sweep measured 5.43 against a quoted 5.44; a second, under
    sustained load an hour later, measured 1.31 — a quarter of it, with every ratio between
    dtypes preserved to two decimal places. Both are true measurements of different clock
    states, and a report that does not say which one it took is not reproducible. So the figure
    is recorded beside the numbers it qualifies.
    """
    fp32 = [r for r in results if r.dtype == "fp32"]
    if not fp32:
        return {}
    best = max(r.tflops for r in fp32)
    return {
        "measured_peak_fp32_tflops": round(best, 2),
        "quoted_peak_fp32_tflops": QUOTED_FP32_TFLOPS,
        "fraction_of_quoted": round(best / QUOTED_FP32_TFLOPS, 3),
        "at_full_clock": best > 0.85 * QUOTED_FP32_TFLOPS,
    }


def _nvidia_smi() -> dict:
    """Clock, temperature and power, straight from the driver.

    Two different runs of this repository's own benchmark differed by 4x and by 2x for reasons
    that turned out to be clock state — once too cold to have ramped, once hot enough to
    throttle. Neither is visible from inside PyTorch. Recording it costs one subprocess and
    makes a saved benchmark say what state the card was in when it was taken.
    """
    import shutil
    import subprocess

    binary = shutil.which("nvidia-smi")
    if not binary:
        return {}
    try:
        out = (
            subprocess.run(
                [
                    binary,
                    "--query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except (subprocess.SubprocessError, OSError, IndexError):
        return {}
    fields = [f.strip() for f in out.split(",")]
    if len(fields) < 3:
        return {}
    return {
        "sm_clock_mhz": fields[0],
        "temperature_c": fields[1],
        "power_w": fields[2],
    }


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
        **_nvidia_smi(),
    }


def save(path: Path, payload: dict) -> Path:
    """Merge into the existing file rather than replacing it.

    `bench --parts matmul` and `bench --parts precision batch` are two runs of the same command
    and both write here. Replacing meant the second silently deleted the first's results, which
    is how the matmul table went missing from an artifact that still looked complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


def as_json(results: list) -> list[dict]:
    return [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in results]
