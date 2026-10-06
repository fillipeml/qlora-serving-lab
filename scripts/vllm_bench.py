"""The continuous-batching comparison, run inside WSL2 because vLLM is Linux only.

The claim continuous batching makes is not that it is faster in aggregate. It is that you stop
paying latency for throughput. This repository's own batch sweep is the thing that claim is made
against: with HuggingFace transformers at fp32, going from batch 1 to batch 16 bought 4.3x the
throughput and cost **17x** the time to first token, because every request in a fixed batch waits
for the whole batch to be scheduled and then for its slowest member to finish.

So this measures both, under the same conditions as that sweep: same model, same dtype, same
prompts, same greedy decoding, same number of new tokens.

**Why `--dtype float32` and not the default.** vLLM resolves `auto` against the model's declared
dtype, sees Qwen2.5 asking for bfloat16, finds a device without it, and falls back to float16 —
which on this card is the path this repository measures at an eighth of fp32. Its log says so
in a line written for exactly this hardware:

    Turing devices tensor cores do not support float32 matmul. To workaround this limitation,
    vLLM will set 'ieee' input precision for chunked prefill triton kernels.

Both dtypes are measured here. The fp16 row is not a mistake to be corrected, it is what anyone
running vLLM on this card without reading the logs actually gets.

**What `gpu_memory_utilization` means, which cost a failed start.** It is a fraction of *total*
VRAM, not of free. On a 6 GiB card with a Windows desktop holding about a gigabyte, 0.85 asks
for 5.1 GiB of the 5.01 GiB that exist and the engine refuses to start — correctly, and with a
message that says exactly that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8300


def prompts(count: int) -> list[str]:
    """The same Portuguese claim notices the rest of the repository is measured on.

    Read from the committed corpus rather than invented here, so the token-length distribution
    matches the HuggingFace sweep exactly. A benchmark on ShareGPT prompts would measure a
    different workload and could not be compared with anything else in this repository.
    """
    import qlora_lab.prompts as p
    from qlora_lab.generate import read

    examples = read(Path("data"), "test")[:count]
    return [p.zero_shot(e.text, e.received_on) for e in examples]


async def one_request(session, prompt: str, max_tokens: int) -> dict:
    """One streaming completion, timed at the first token and at the last.

    Streaming is not a detail: without it there is no time to first token to report, and the
    whole claim being tested is about latency rather than aggregate throughput.
    """
    started = time.perf_counter()
    first: float | None = None
    produced = 0
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    async with session.post(f"http://127.0.0.1:{PORT}/v1/completions", json=payload) as response:
        async for raw in response.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            body = line[len("data: ") :]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            text = chunk["choices"][0].get("text", "")
            if text:
                if first is None:
                    first = time.perf_counter()
                produced += 1
    finished = time.perf_counter()
    return {
        "ttft_ms": ((first or finished) - started) * 1000,
        "total_ms": (finished - started) * 1000,
        "tokens": produced,
    }


async def run_concurrency(level: int, texts: list[str], max_tokens: int) -> dict:
    """Fire `level` requests at once and keep `level` in flight until all are done.

    This is the comparison with a fixed batch of the same size: the same number of requests are
    resident in the engine, and the difference is only whether they must start and finish
    together.
    """
    import aiohttp

    queue = asyncio.Queue()
    for text in texts:
        queue.put_nowait(text)

    results: list[dict] = []
    began = time.perf_counter()

    async with aiohttp.ClientSession() as session:

        async def worker() -> None:
            while True:
                try:
                    text = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results.append(await one_request(session, text, max_tokens))

        await asyncio.gather(*[worker() for _ in range(level)])

    elapsed = time.perf_counter() - began
    ttfts = sorted(r["ttft_ms"] for r in results)
    return {
        "concurrency": level,
        "requests": len(results),
        "seconds": round(elapsed, 2),
        "records_per_second": round(len(results) / elapsed, 3),
        "ttft_p50_ms": round(statistics.median(ttfts), 1),
        "ttft_p95_ms": round(ttfts[int(0.95 * (len(ttfts) - 1))], 1),
        "ttft_max_ms": round(ttfts[-1], 1),
        "output_tokens_per_second": round(sum(r["tokens"] for r in results) / elapsed, 1),
    }


def serve(dtype: str, max_num_seqs: int, utilisation: float, eager: bool) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--dtype",
        dtype,
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(utilisation),
        "--port",
        str(PORT),
    ]
    if eager:
        command.append("--enforce-eager")
    # To a file, not a pipe. vLLM runs its engine in a separate process whose output does not
    # come back through the parent's pipe, so a failed start produced an empty log and a
    # benchmark that could not say why its subject died. A file catches everything, including
    # the grandchild's.
    # noqa: SIM115 — the handle must outlive this function; the server writes to it until it is
    # terminated, and closing it here would give the child a closed descriptor.
    log = Path("server.log").open("w", encoding="utf-8")  # noqa: SIM115
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)


def wait_for_server(process: subprocess.Popen, timeout: float = 1200) -> list[str]:
    """Block until the server answers, and keep the tail of its log either way.

    The lines kept are the evidence for which attention backend was chosen, what dtype was
    resolved, and how much key-value cache fitted — which is what makes one run comparable with
    another, and what says whether this card took the path the report claims.

    First start is slow: Triton compiles its kernels at run time, and on a fresh machine that is
    minutes before the server answers. Later runs hit its cache.
    """
    import urllib.error
    import urllib.request

    log = Path("server.log")
    keys = (
        "attention backend",
        "KV cache size",
        "Turing",
        "Upcasting",
        "Maximum concurrency",
        "FA version",
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            raise RuntimeError("the server exited before answering:" + chr(10) + chr(10).join(tail))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as reply:
                if reply.status == 200:
                    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                    return [line.strip() for line in lines if any(k in line for k in keys)]
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    raise TimeoutError("the server did not answer in time:" + chr(10) + chr(10).join(tail))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--out", default="results/vllm.json")
    args = parser.parse_args()

    texts = prompts(args.requests)
    print(f"{len(texts)} Portuguese claim prompts, {args.dtype}, max_num_seqs={args.max_num_seqs}")

    process = serve(args.dtype, args.max_num_seqs, args.gpu_memory_utilization, args.enforce_eager)
    try:
        log = wait_for_server(process)
        print("server up. what it chose:")
        for line in log:
            print("  " + line[:160])

        # Warm up at the highest level before measuring anything. Triton compiles its kernels
        # at run time and specialises them by shape, so the first requests at a new concurrency
        # pay for a compile that later ones do not. Measured without this, the fp16 sweep put
        # concurrency 2 at 0.287 records per second against concurrency 1's 0.785 — a curve
        # going the wrong way, which is the shape of a measurement artefact rather than a
        # property of the engine.
        warm = max(args.levels)
        print(f"  warming up at concurrency {warm} (discarded)", flush=True)
        asyncio.run(run_concurrency(warm, texts[: min(len(texts), warm * 2)], args.max_tokens))

        rows = []
        for level in args.levels:
            row = asyncio.run(run_concurrency(level, texts, args.max_tokens))
            rows.append(row)
            print(
                f"  concurrency {row['concurrency']:3d}  {row['records_per_second']:6.3f} rec/s  "
                f"TTFT p50 {row['ttft_p50_ms']:8.1f} ms  p95 {row['ttft_p95_ms']:8.1f} ms  "
                f"{row['output_tokens_per_second']:7.1f} tok/s",
                flush=True,
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing[args.dtype] = {
        "model": MODEL,
        "dtype": args.dtype,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "max_tokens": args.max_tokens,
        "server_log": log,
        "levels": rows,
        "environment": {
            "platform": "WSL2 Ubuntu 26.04 on Windows 11",
            "cc": os.environ.get("CC", ""),
        },
    }
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
