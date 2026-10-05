"""QLoRA on one 6 GiB consumer card, with the parts the recipes get wrong on this hardware.

Four decisions here are not the default, and each one is a measurement rather than a preference:

1. **The loss is computed on the answer only.** The prompt tokens are masked to -100. Training
   on the prompt too teaches the model to reproduce messages it will never be asked to write,
   and on a 1,000-row set it is most of the gradient. This is the single change with the largest
   effect in the ablation.

2. **fp16 autocast, not bf16.** Every QLoRA recipe branches on
   `torch.cuda.is_bf16_supported()`, which returns True on this card by counting software
   emulation. The branch therefore fires, and the resulting bf16 path runs on emulated kernels.
   What makes this worth measuring rather than asserting: this card's *native* fp16 GEMM is
   worse still. See the README; the conclusion is counter-intuitive and the numbers are in the
   repository.

3. **fp16 needs its gradient scaler, and QLoRA makes that sharper.** Gradients in fp16
   underflow to zero below about 6e-8. The scaler multiplies the loss up, unscales before the
   step, and skips any step whose gradients went to infinity. Skipped steps are counted and
   reported: a run that silently skips a third of its steps has trained on two thirds of the
   data it claims.

4. **The adapter is on every projection, not just q and v.** The original LoRA paper attached to
   the attention projections because that was what fitted. At rank 16 on a 1.5B model the
   adapter is small enough that the MLP projections fit too, and the ablation says they matter
   for this task — the format is learned by the MLP, not by attention.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .generate import Example
from .prompts import training_pair

#: Attention *and* MLP. Qwen's names; a different family needs a different list, which is why it
#: is a constant to be edited rather than a string match that silently attaches to nothing.
TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class Settings:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output: str = "adapters/qwen2.5-1.5b-nf4-r16"
    rank: int = 16
    #: Two times the rank. The ratio matters more than either number: alpha/rank is the scale
    #: the adapter's output is multiplied by, so raising the rank without raising alpha quietly
    #: weakens every adapter you add.
    alpha: int = 32
    dropout: float = 0.05
    epochs: float = 3.0
    learning_rate: float = 2e-4
    #: One, because the card has 6 GiB. The effective batch is this times the accumulation
    #: steps, and only the effective batch affects the result.
    #: Four, not one. The usual advice for a 6 GiB card is a batch of one and a long
    #: accumulation, but the measured peak here is 2.05 GiB at batch 1 — the card is two thirds
    #: idle, and the kernels are far from saturated. Raising the batch and lowering the
    #: accumulation to match keeps the effective batch identical and cuts the wall clock.
    batch_size: int = 4
    accumulation: int = 2
    max_length: int = 512
    seed: int = 20261005
    #: Paged, so a gradient spike that would otherwise be an out-of-memory error moves optimiser
    #: state to host RAM instead of killing a run that is forty minutes in.
    optimiser: str = "paged_adamw_8bit"
    precision: str = "nf4-dq"
    #: Which autocast the forward pass runs under: "fp16", "bf16" or "off".
    #:
    #: Not a style question on this card. fp16 autocast sends every matmul to the kernel
    #: `bench.py` measures at a ninth of fp32, so the default recipe is the slow one here. "off"
    #: keeps the dequantised fp32 compute and needs no gradient scaler at all. The three are
    #: timed against each other in the README rather than argued about.
    autocast: str = "off"
    train_on_prompt: bool = False
    log_every: int = 25


@dataclass
class Progress:
    """What a run did, written beside the adapter so a result can be traced to the run.

    `skipped_steps` is here because fp16 training can lose steps to overflow without saying so,
    and a loss curve looks fine while it happens.
    """

    settings: dict
    steps: int
    skipped_steps: int
    final_loss: float
    best_eval_loss: float
    seconds: float
    peak_vram_bytes: int
    losses: list[tuple[int, float]]
    eval_losses: list[tuple[int, float]]
    bf16_hardware: bool
    device: str


def encode(example: Example, tokenizer, max_length: int, train_on_prompt: bool) -> dict:
    """One training row, with the prompt masked out of the loss.

    Tokenised in two pieces rather than one, because the boundary has to be known exactly. Doing
    it in one pass and then finding the prompt's length by string search is where this goes
    wrong: the tokeniser can merge the last character of the prompt with the first of the
    answer, and the mask then covers one token too many or too few — silently, and only on some
    rows.
    """
    prompt, target = training_pair(example)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = list(input_ids)
    if not train_on_prompt:
        for index in range(min(len(prompt_ids), len(labels))):
            labels[index] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def collate(rows: list[dict], pad_id: int) -> dict:
    """Pad right, and pad the labels with -100 rather than with the pad token.

    Padding the labels with the pad token would train the model to emit padding, which is both
    wrong and invisible: the loss goes down, because predicting padding is easy.
    """
    width = max(len(r["input_ids"]) for r in rows)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in rows:
        gap = width - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_id] * gap)
        batch["attention_mask"].append(row["attention_mask"] + [0] * gap)
        batch["labels"].append(row["labels"] + [-100] * gap)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def _autocast(mode: str):
    """The context the forward pass runs in.

    `nullcontext` rather than `autocast(enabled=False)` because the two are not the same: the
    second still enters the autocast machinery and still costs a cast at every boundary.
    """
    if mode == "off":
        return nullcontext()
    return torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[mode])


@torch.no_grad()
def evaluate(
    model, rows: list[dict], pad_id: int, batch_size: int = 2, autocast: str = "off"
) -> float:
    """Mean loss over a split, in the same autocast the training step uses."""
    model.eval()
    total = 0.0
    batches = 0
    for start in range(0, len(rows), batch_size):
        batch = collate(rows[start : start + batch_size], pad_id)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with _autocast(autocast):
            total += model(**batch).loss.detach().item()
        batches += 1
    model.train()
    return total / max(batches, 1)


def train(
    settings: Settings,
    train_examples: list[Example],
    eval_examples: list[Example],
) -> Progress:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from .infer import bf16_is_real, load

    torch.manual_seed(settings.seed)
    loaded = load(settings.model_id, settings.precision)
    tokenizer = loaded.tokenizer
    model = loaded.model

    # Casts the layer norms and the head to fp32 and enables gradient checkpointing. Without
    # the fp32 norms a 4-bit model's loss goes to NaN within a few dozen steps; without the
    # checkpointing the activations for a 512-token sequence do not fit beside the weights.
    # `use_reentrant=False` explicitly: the old implementation is deprecated, and it is also
    # the one that silently drops gradients when some inputs do not require them — which is
    # every QLoRA setup, where the base weights are frozen.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=settings.rank,
            lora_alpha=settings.alpha,
            lora_dropout=settings.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(TARGET_MODULES),
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    rows = [
        encode(e, tokenizer, settings.max_length, settings.train_on_prompt) for e in train_examples
    ]
    eval_rows = [
        encode(e, tokenizer, settings.max_length, settings.train_on_prompt) for e in eval_examples
    ]
    pad_id = tokenizer.pad_token_id

    import bitsandbytes as bnb

    optimiser_class = {
        "paged_adamw_8bit": bnb.optim.PagedAdamW8bit,
        "adamw_8bit": bnb.optim.AdamW8bit,
        "adamw": torch.optim.AdamW,
    }[settings.optimiser]
    optimiser = optimiser_class(
        [p for p in model.parameters() if p.requires_grad], lr=settings.learning_rate
    )

    steps_per_epoch = math.ceil(len(rows) / (settings.batch_size * settings.accumulation))
    total_steps = int(steps_per_epoch * settings.epochs)
    warmup = max(1, int(0.03 * total_steps))

    def learning_rate_at(step: int) -> float:
        if step < warmup:
            return settings.learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return settings.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    # Only fp16 underflows: bf16 has fp32's exponent range and fp32 is fp32. Enabling the
    # scaler anyway would be harmless but dishonest, because `skipped_steps` would then be
    # reported for configurations that cannot skip.
    scaler = torch.amp.GradScaler("cuda", enabled=settings.autocast == "fp16")
    generator = torch.Generator().manual_seed(settings.seed)

    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    losses: list[tuple[int, float]] = []
    eval_losses: list[tuple[int, float]] = []
    best_eval = float("inf")
    skipped = 0
    step = 0
    running = 0.0
    micro = 0

    model.train()
    print(
        f"{settings.model_id} at {settings.precision} | adapter {trainable:,} of "
        f"{total_params:,} parameters ({trainable / total_params:.2%}) | "
        f"{total_steps} optimiser steps | autocast {settings.autocast} | "
        f"bf16 in hardware: {bf16_is_real()}"
    )

    for _ in range(math.ceil(settings.epochs)):
        order = torch.randperm(len(rows), generator=generator).tolist()
        for position in range(0, len(order), settings.batch_size):
            if step >= total_steps:
                break
            indices = order[position : position + settings.batch_size]
            batch = collate([rows[i] for i in indices], pad_id)
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}

            with _autocast(settings.autocast):
                loss = model(**batch).loss / settings.accumulation
            scaler.scale(loss).backward()
            running += loss.detach().item() * settings.accumulation
            micro += 1

            if micro % settings.accumulation:
                continue

            for group in optimiser.param_groups:
                group["lr"] = learning_rate_at(step)
            if scaler.is_enabled():
                scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            before = scaler.get_scale() if scaler.is_enabled() else 0.0
            scaler.step(optimiser)
            scaler.update()
            # A reduced scale means the step was skipped because a gradient overflowed. Nothing
            # else reports it, and a run that skips many steps has trained on less data than it
            # says.
            if scaler.is_enabled() and scaler.get_scale() < before:
                skipped += 1
            optimiser.zero_grad(set_to_none=True)

            step += 1
            mean = running / (settings.accumulation * settings.batch_size)
            losses.append((step, mean))
            running = 0.0

            if step % settings.log_every == 0 or step == total_steps:
                value = evaluate(model, eval_rows, pad_id, autocast=settings.autocast)
                eval_losses.append((step, value))
                best_eval = min(best_eval, value)
                print(
                    f"  step {step:4d}/{total_steps}  train {mean:.4f}  eval {value:.4f}  "
                    f"lr {learning_rate_at(step):.2e}  skipped {skipped}  "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                )
        if step >= total_steps:
            break

    seconds = time.perf_counter() - began
    directory = Path(settings.output)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))

    progress = Progress(
        settings=asdict(settings),
        steps=step,
        skipped_steps=skipped,
        final_loss=losses[-1][1] if losses else float("nan"),
        best_eval_loss=best_eval,
        seconds=seconds,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        losses=losses,
        eval_losses=eval_losses,
        bf16_hardware=bf16_is_real(),
        device=torch.cuda.get_device_name(0),
    )
    (directory / "run.json").write_text(json.dumps(asdict(progress), indent=2), encoding="utf-8")
    return progress
