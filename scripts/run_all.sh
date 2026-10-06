#!/usr/bin/env bash
# Every number in the README, in the order they were produced.
#
# Run from the repository root with the `gpu` extra installed. On a GTX 1660 Ti this is about
# two and a half hours, most of it in section 4.
#
#   bash scripts/run_all.sh 2>&1 | tee results/run.log
#
# The test split is touched once, in section 7, after every decision has been made. Validation
# is used during training and for one sanity check; no threshold, prompt or hyper-parameter in
# this repository was chosen by looking at a test number.

set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
ADAPTER=${ADAPTER:-adapters/qwen2.5-0.5b-r16}
SHORT=$(basename "$MODEL")

echo "=== 1. the corpus ==="
$PY -m qlora_lab data

echo
echo "=== 2. the hardware, before anything is trained ==="
# Twenty seconds, and the rest of the configuration rests on it.
$PY -m qlora_lab bench --parts matmul --model "$MODEL"

echo
echo "=== 3. the baseline with no model at all ==="
$PY -m qlora_lab baseline --split validation

echo
echo "=== 4. QLoRA ==="
# Two epochs at an effective batch of 8, which is 250 optimiser steps. The first attempt used
# an effective batch of 32 and got 32 steps out of one epoch: enough to learn the output format
# perfectly (JSON validity went from 1.3% to 100%) and nowhere near enough to learn the
# content (8% of records exactly right). The number of steps was the binding constraint, not
# the number of rows.
$PY -m qlora_lab train --model "$MODEL" --output "$ADAPTER" --epochs 2 --accumulation 2

echo
echo "=== 5. the adapter answers at all ==="
$PY -m qlora_lab tuned --split validation --limit 50 --model "$MODEL" --adapter "$ADAPTER"

echo
echo "=== 6. serving: six precisions, then batch size ==="
$PY -m qlora_lab bench --parts precision batch --model "$MODEL" --split validation --limit 16

echo
echo "=== 7. the test split, once ==="
$PY -m qlora_lab baseline --split test
# Batched, and batching is safe: the outputs are identical to batch 1 on every record checked,
# greedy decoding being deterministic. It is also 3.6x faster on this model, which is the
# difference between running the table and not running it. The latency column in the README
# comes from the batch-1 sweep in section 6, never from these runs.
$PY -m qlora_lab prompted --split test --model "$MODEL" --styles zero-shot few-shot --batch-size 8
$PY -m qlora_lab tuned --split test --model "$MODEL" --adapter "$ADAPTER" --batch-size 8

echo
echo "=== 8. the held-out vocabulary, which is the actual question ==="
# The rules and the generator share an author, and the adapter was trained on the same
# templates the test split is drawn from. Both scores above are optimistic and they are
# optimistic in ways that do not cancel. These 200 records use different words for the same
# eight perils, different date forms, different phrasings for every boolean, and twenty-four
# municipalities that are not in the rule set's gazetteer.
$PY -m qlora_lab baseline --split shifted
$PY -m qlora_lab prompted --split shifted --model "$MODEL" --styles zero-shot few-shot --batch-size 8
$PY -m qlora_lab tuned --split shifted --model "$MODEL" --adapter "$ADAPTER" --batch-size 8

echo
echo "=== 9. the paired comparisons ==="
$PY -m qlora_lab compare \
  "rules.test.json" \
  "zero-shot-${SHORT}-nf4.test.json" \
  "few-shot-${SHORT}-nf4.test.json" \
  "tuned-${SHORT}-nf4.test.json"
