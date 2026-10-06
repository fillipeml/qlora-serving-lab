"""The commands that produce every number in the README.

    python -m qlora_lab data                 write the splits
    python -m qlora_lab baseline             rules, on the split you name
    python -m qlora_lab prompted             zero-shot and few-shot
    python -m qlora_lab train                QLoRA
    python -m qlora_lab tuned                the adapter, scored
    python -m qlora_lab bench                matmul, precision and batch sweeps
    python -m qlora_lab compare              the paired tests between saved runs

Everything writes to `results/` and nothing reads the test split unless asked. The default split
is `validation` on every command for that reason: a test number should be produced on purpose,
once, and the way to make that true is to make it the inconvenient option.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import Outcome, run_model, run_rules
from .generate import generate, read, write

DATA = Path("data")
RESULTS = Path("results")
# 0.5B, not 1.5B. Both fit; the smaller one is the honest subject for a report about one
# consumer card, because it is the size at which the whole table — four systems, 300 records,
# six precisions and a batch sweep — can actually be run and re-run. The 1.5B appears once, as
# a scaling row, measured the same way.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _load(split: str, limit: int | None) -> list:
    examples = read(DATA, split)
    return examples[:limit] if limit else examples


def _show(outcome: Outcome) -> None:
    print()
    print("\n".join(outcome.report.lines()))
    qualifier = "" if outcome.batch_size == 1 else f" at batch {outcome.batch_size}"
    print(
        f"  cost                 {outcome.median_ms:.1f} ms/record (median){qualifier}, "
        f"TTFT {outcome.median_ttft_ms:.1f} ms, {outcome.decode_tokens_per_second:.1f} tok/s, "
        f"{outcome.prompt_tokens} prompt tokens, {outcome.weight_gib:.2f} GiB of weights"
    )
    print(f"  saved to {outcome.save(RESULTS)}")


def cmd_data(args: argparse.Namespace) -> int:
    from .shift import generate_shifted

    # The shifted set is written beside the others and is a test split like any other, except
    # that no system in this repository has seen a word of its vocabulary.
    counts = write(DATA, generate(seed=args.seed) + generate_shifted())
    print(f"wrote {counts} to {DATA}/")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    _show(run_rules(_load(args.split, args.limit), args.split))
    return 0


def cmd_prompted(args: argparse.Namespace) -> int:
    examples = _load(args.split, args.limit)
    pool = read(DATA, "train")
    for style in args.styles:
        _show(
            run_model(
                examples,
                args.split,
                args.model,
                style,
                precision=args.precision,
                shot_pool=pool,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import Settings, train

    settings = Settings(
        model_id=args.model,
        output=args.output,
        rank=args.rank,
        alpha=args.alpha,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        accumulation=args.accumulation,
        precision=args.precision,
        autocast=args.autocast,
        train_on_prompt=args.train_on_prompt,
    )
    if args.max_steps:
        # An optimiser step consumes batch_size * accumulation rows, and leaving the batch out
        # of this is how a "40-step timing run" silently became a 10-step one.
        rows_per_step = settings.batch_size * settings.accumulation
        settings.epochs = args.max_steps * rows_per_step / len(read(DATA, "train"))
    progress = train(settings, read(DATA, "train"), read(DATA, "validation")[: args.eval_size])
    print(
        f"\n{progress.steps} steps in {progress.seconds / 60:.1f} min, "
        f"{progress.skipped_steps} skipped to gradient overflow, "
        f"best eval loss {progress.best_eval_loss:.4f}, "
        f"peak {progress.peak_vram_bytes / 2**30:.2f} GiB"
    )
    return 0


def cmd_tuned(args: argparse.Namespace) -> int:
    _show(
        run_model(
            _load(args.split, args.limit),
            args.split,
            args.model,
            "tuned",
            precision=args.precision,
            adapter=args.adapter,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from . import bench, prompts

    payload: dict = {"device": bench.describe_device()}
    print(json.dumps(payload["device"], indent=2))

    if "matmul" in args.parts:
        results = bench.matmul_sweep()
        payload["matmul"] = bench.as_json(results)
        payload["bandwidth"] = bench.as_json(bench.bandwidth_check())
        payload["clock"] = bench.clock_health(results)
        print(
            "\nclock: fp32 peaked at "
            f"{payload['clock']['measured_peak_fp32_tflops']} TFLOP/s, "
            f"{payload['clock']['fraction_of_quoted']:.0%} of the card's quoted "
            f"{payload['clock']['quoted_peak_fp32_tflops']}"
            + ("" if payload["clock"]["at_full_clock"] else "  <- THROTTLED, numbers are low")
        )
        print("\nmatrix multiply, TFLOP/s")
        for row in payload["matmul"]:
            print(f"  {row['dtype']:14s} n={row['size']:5d}  {row['tflops']:6.2f}")
        print("element-wise, GB/s")
        for row in payload["bandwidth"]:
            print(f"  {row['dtype']:14s}            {row['tflops']:6.1f}")

    examples = _load(args.split, args.limit or 16)
    texts = [prompts.zero_shot(e.text, e.received_on) for e in examples]

    if "precision" in args.parts:
        payload["serving"] = bench.as_json(bench.serving_sweep(args.model, texts))
        print("\nthe same model at six precisions")
        print(f"  {'precision':10s} {'weights':>9s} {'peak':>7s} {'TTFT':>9s} {'decode':>11s}")
        for row in payload["serving"]:
            print(
                f"  {row['precision']:10s} {row['weight_gib']:7.2f}Gi {row['peak_gib']:6.2f}Gi "
                f"{row['ttft_ms']:7.1f}ms {row['decode_tokens_per_second']:8.1f}t/s"
            )

    if "compute" in args.parts:
        payload["compute_dtype"] = bench.compute_dtype_sweep(args.model, texts)
        print("\nthe same 4-bit weights, dequantised into three dtypes")
        for row in payload["compute_dtype"]:
            print(
                f"  compute={row['compute_dtype']:5s} {row['weight_gib']:5.2f}Gi "
                f"TTFT {row['ttft_ms']:8.1f}ms  decode {row['decode_tokens_per_second']:6.1f} t/s"
            )

    if "batch" in args.parts:
        payload["batch"] = bench.batch_sweep(args.model, texts, precision=args.precision)
        print("\nbatch size")
        for row in payload["batch"]:
            if "error" in row:
                print(f"  {row['batch_size']:3d}  {row['error']}")
            else:
                print(
                    f"  {row['batch_size']:3d}  {row['records_per_second']:6.2f} rec/s  "
                    f"TTFT {row['ttft_ms']:7.1f} ms  peak {row['peak_gib']:.2f} GiB"
                )

    print(f"\nsaved to {bench.save(RESULTS / 'bench.json', payload)}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Paired tests between saved runs, without re-running anything.

    The saved outputs carry `exact_by_record` for exactly this: a comparison between two systems
    that ran hours apart is still paired, because both answered the same records in the same
    order.
    """
    from .stats import sign_test

    loaded = []
    for name in args.runs:
        path = RESULTS / name if (RESULTS / name).exists() else Path(name)
        loaded.append(json.loads(path.read_text(encoding="utf-8")))

    width = max(len(p["system"]) for p in loaded)
    print(f"{'system':{width}s} {'exact record':>24s} {'ms/record':>10s}")
    for payload in loaded:
        print(
            f"{payload['system']:{width}s} {payload['scores']['exact_record']:>24s} "
            f"{payload['median_ms']:9.1f}"
        )

    splits = {p["split"] for p in loaded}
    if len(splits) > 1:
        print(f"\nnot comparing: these runs are on different splits ({sorted(splits)})")
        return 1

    print()
    for left, right in zip(loaded, loaded[1:], strict=False):
        paired = sign_test(left["exact_by_record"], right["exact_by_record"])
        verdict = "significant at 0.05" if paired.p_value < 0.05 else "not significant at 0.05"
        print(f"{left['system']} -> {right['system']}: {paired} ({verdict})")
    return 0


def cmd_hybrid(args: argparse.Namespace) -> int:
    """Compose a saved model run with the rule extractor. No GPU, no model, no re-running."""
    import time

    from .evaluate import Outcome
    from .hybrid import compose_outputs, score

    path = RESULTS / args.run if (RESULTS / args.run).exists() else Path(args.run)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["split"] != args.split:
        print(f"{path.name} is the {payload['split']} split, not {args.split}")
        return 1

    examples = read(DATA, args.split)
    started = time.perf_counter()
    composed = compose_outputs(examples, payload["outputs"])
    rules_ms = (time.perf_counter() - started) * 1000 / max(len(examples), 1)
    name = f"hybrid:{payload['system']}"

    _show(
        Outcome(
            system=name,
            split=args.split,
            outputs=composed,
            report=score(name, examples, payload["outputs"]),
            # The model's cost plus the rule extractor's, which is the honest total: the hybrid
            # runs both, and the second one is free only in the sense that a millisecond is.
            median_ms=payload["median_ms"] + rules_ms,
            median_ttft_ms=payload["median_ttft_ms"],
            decode_tokens_per_second=payload["decode_tokens_per_second"],
            prompt_tokens=payload["prompt_tokens"],
            weight_gib=payload["weight_gib"],
            batch_size=payload.get("batch_size", 1),
            notes=f"composed from {path.name}; arithmetic fields from the rule extractor",
        )
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """One record through every saved system, from the committed results and no GPU."""
    from .show import describe, load_runs, pick_disagreement, prepare

    examples = prepare(DATA, args.split)
    runs = load_runs(RESULTS, args.split)
    if not runs:
        print(f"no saved runs for the {args.split} split; run the evaluations first")
        return 1

    if args.record:
        chosen = next((e for e in examples if e.id == args.record), None)
        if chosen is None:
            print(f"no record {args.record} in the {args.split} split")
            return 1
    else:
        chosen = pick_disagreement(examples, runs) or examples[0]

    print("\n".join(describe(chosen, runs)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qlora_lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        # `validation` by default on purpose: see the module docstring.
        p.add_argument(
            "--split",
            default="validation",
            choices=("train", "validation", "test", "shifted"),
        )
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--precision", default="nf4")
        p.add_argument("--batch-size", type=int, default=1)
        # Generous by default. A cap that binds is not a measurement of the model, it is a
        # measurement of the cap: at 110 tokens the base model's pretty-printed, fenced object
        # was cut mid-record and 296 of 300 outputs scored as invalid JSON for a reason that
        # had nothing to do with the model.
        p.add_argument("--max-new-tokens", type=int, default=256)

    p = sub.add_parser("data", help="write the splits")
    p.add_argument("--seed", type=int, default=20261005)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("baseline", help="rules, no model")
    common(p)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("prompted", help="zero-shot and few-shot")
    common(p)
    p.add_argument("--styles", nargs="+", default=["zero-shot", "few-shot"])
    p.set_defaults(func=cmd_prompted)

    p = sub.add_parser("train", help="QLoRA")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default="adapters/run")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    # 2, matching Settings. With a batch of 4 that is an effective batch of 8 and 125
    # optimiser steps per epoch. At accumulation 8 one epoch is 32 steps, which is too few for
    # the adapter to converge — measured: the format is learned, the content is not.
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--precision", default="nf4-dq")
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--autocast", default="off", choices=("off", "fp16", "bf16"))
    p.add_argument("--max-steps", type=int, default=0, help="stop early; for timing runs")
    p.add_argument(
        "--train-on-prompt",
        action="store_true",
        help="the ablation: do not mask the prompt out of the loss",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("tuned", help="score an adapter")
    common(p)
    p.add_argument("--adapter", required=True)
    p.set_defaults(func=cmd_tuned)

    p = sub.add_parser("bench", help="hardware and serving benchmarks")
    common(p)
    p.add_argument(
        "--parts",
        nargs="+",
        default=["matmul", "precision", "compute", "batch"],
        choices=("matmul", "precision", "compute", "batch"),
    )
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("hybrid", help="compose a saved model run with the rule extractor")
    p.add_argument("run", help="a saved model run, e.g. tuned-Qwen2.5-0.5B-Instruct-nf4.test.json")
    p.add_argument("--split", default="test", choices=("train", "validation", "test", "shifted"))
    p.set_defaults(func=cmd_hybrid)

    p = sub.add_parser("show", help="one record through every saved system")
    p.add_argument("--split", default="test", choices=("train", "validation", "test", "shifted"))
    p.add_argument("--record", default=None, help="a record id; omitted, picks a disagreement")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("compare", help="paired tests between saved runs")
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Every message in the corpus is Portuguese, and a Windows console defaults to a codepage
    # that cannot print it: the text comes out as mojibake and a redirect to a file raises
    # UnicodeEncodeError outright. Reconfiguring here rather than asking the reader to set
    # PYTHONIOENCODING, because a tool that cannot print its own data is broken.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
