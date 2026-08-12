# Benchmarks

Runnable latency and accuracy benchmark scripts for user-supplied model artifacts and datasets.

> **Evidence boundary:** the two JSON files currently committed under `results/` are hand-authored, fictional planning fixtures. They were not emitted by these scripts, were not collected on the named hardware, and are **not measured benchmark evidence**. Do not cite their values as model or device performance. Only output produced by running the commands below with a documented checkpoint, dataset, environment, and command is empirical evidence.

## Scripts

- [`latency_benchmark.py`](./latency_benchmark.py) — measures p50 / p95 / p99 inference latency across batch sizes on the current device (CPU, GPU, or Jetson, auto-detected).
- [`accuracy_benchmark.py`](./accuracy_benchmark.py) — per-class precision, recall, mAP, and confusion matrix against a labeled test set.

## Illustrative fixtures (committed, not measured)

- [`results/cpu-vs-gpu-vs-jetson.json`](./results/cpu-vs-gpu-vs-jetson.json) — fictional latency values used to discuss hardware trade-offs.
- [`results/class-balance-impact.json`](./results/class-balance-impact.json) — fictional class-balance values used to discuss evaluation design.

No checkpoint, source dataset, raw timing log, runner output, or hardware record supports these numbers. They make no promise about absolute **or relative** performance.

## Running

### Latency

```bash
# On the machine where Triton or a local engine is available:
python benchmarks/latency_benchmark.py \
    --model-path checkpoints/best.pt \
    --batch-sizes 1 2 4 8 16 \
    --warmup 50 \
    --iterations 500 \
    --output benchmarks/results/my-machine-latency.json
```

### Accuracy

```bash
python benchmarks/accuracy_benchmark.py \
    --model-path checkpoints/best.pt \
    --data-yaml data/weld_defect.yaml \
    --split test \
    --output benchmarks/results/my-accuracy.json
```

## Interpreting output

Do not carry a latency or accuracy number from one device, precision, model export, thermal state, or dataset to another. Record the model checksum, dataset provenance, dependency versions, full command, device details, warmup count, and raw samples with any measured report. The committed fixtures may help design a test matrix, but they are not acceptance thresholds.
