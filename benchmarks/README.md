# Benchmarks

Latency and accuracy benchmarks for the weld-defect model across hardware targets and defect classes.

## Scripts

- [`latency_benchmark.py`](./latency_benchmark.py) — measures p50 / p95 / p99 inference latency across batch sizes on the current device (CPU, GPU, or Jetson, auto-detected).
- [`accuracy_benchmark.py`](./accuracy_benchmark.py) — per-class precision, recall, mAP, and confusion matrix against a labeled test set.

## Sample results (committed)

- [`results/cpu-vs-gpu-vs-jetson.json`](./results/cpu-vs-gpu-vs-jetson.json) — latency across three hardware targets and three precisions.
- [`results/class-balance-impact.json`](./results/class-balance-impact.json) — how per-class precision and recall change with different class-balancing strategies at training time.

These results are illustrative of real engagement shapes; reproducing on your hardware will give similar relative numbers but will vary absolutely.

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

## Expectations

For the Jetson Orin AGX targets described in [`docs/production/edge-deployment.md`](../docs/production/edge-deployment.md), expect:

| Device | Precision | Batch 1 P95 (ms) | Batch 8 P95 (ms) |
|---|---|---|---|
| CPU (x86 16-core) | FP32 | 120 | 840 |
| NVIDIA L4 (AWS g6) | FP16 | 12 | 26 |
| NVIDIA A100 (bare metal) | FP16 | 6 | 18 |
| Jetson Orin AGX 64 GB | INT8 | 10 | 34 |
| Jetson Orin NX 16 GB | INT8 | 18 | 62 |

See the [`results/`](./results/) JSON files for full breakdowns.
