#!/usr/bin/env bash
# Every number in the README, in the order they were produced.
#
# Run from the repository root with the `gpu` extra installed. Expect two to three hours on a
# GTX 1660 Ti. Each step writes to results/ and nothing reads the test split until the last
# section, which is the point: development happens on validation, and the test split is touched
# once, on purpose, at the end.
#
#   bash scripts/run_all.sh 2>&1 | tee results/run.log

set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
ADAPTER=${ADAPTER:-adapters/qwen2.5-1.5b-r16}

echo "=== 1. the corpus ==="
$PY -m qlora_lab data

echo
echo "=== 2. the hardware, before anything is trained ==="
# The matmul sweep is twenty seconds and it is what the rest of the configuration rests on.
$PY -m qlora_lab bench --parts matmul --model "$MODEL"

echo
echo "=== 3. the baseline with no model at all ==="
$PY -m qlora_lab baseline --split validation

echo
echo "=== 4. the base model, prompted ==="
$PY -m qlora_lab prompted --split validation --model "$MODEL" --styles zero-shot few-shot

echo
echo "=== 5. QLoRA ==="
$PY -m qlora_lab train --model "$MODEL" --output "$ADAPTER" --epochs 3

echo
echo "=== 6. the adapter, scored ==="
$PY -m qlora_lab tuned --split validation --model "$MODEL" --adapter "$ADAPTER"

echo
echo "=== 7. serving: six precisions, then batch size ==="
$PY -m qlora_lab bench --parts precision batch --model "$MODEL" --split validation --limit 16

echo
echo "=== 8. the test split, once ==="
$PY -m qlora_lab baseline --split test
$PY -m qlora_lab prompted --split test --model "$MODEL" --styles zero-shot few-shot
$PY -m qlora_lab tuned --split test --model "$MODEL" --adapter "$ADAPTER"

echo
echo "=== 9. the paired comparisons ==="
$PY -m qlora_lab compare \
  rules.test.json \
  "zero-shot-Qwen2.5-1.5B-Instruct-nf4.test.json" \
  "few-shot-Qwen2.5-1.5B-Instruct-nf4.test.json" \
  "tuned-Qwen2.5-1.5B-Instruct-nf4.test.json"
