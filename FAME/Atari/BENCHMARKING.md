# Atari Continual Learning Benchmarking

Reproduces experiments from: "Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning" (ICLR 2026)

## Quick Test

```bash
cd Atari
./benchmark_experiments.sh --quick
```

Runs Freeway with 10K timesteps, 1 seed (~5 minutes).

## Full Reproduction

```bash
./benchmark_experiments.sh
```

Runs all algorithms on both games with 3 seeds and 1M timesteps per task (~135 hours on a single GPU).

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--env=ENV` | Game(s), comma-separated | Both games |
| `--seeds=1,2,3` | Seeds | `1,2,3` |
| `--timesteps=N` | Steps per task | `1000000` |
| `--algorithms=LIST` | Algorithms to run | All 6 |
| `--skip-training` | Skip to evaluation | Off |
| `--skip-eval` | Skip evaluation | Off |
| `--skip-process` | Skip results processing | Off |
| `--quick` | Quick test mode | Off |

## Algorithms

| CLI Name | Paper Name | Description |
|----------|-----------|-------------|
| `cnn-simple` | Baseline | Train from scratch |
| `cnn-simple-ft` | FT | Fine-tuning |
| `cnn-componet` | CompoNet | Compositional network |
| `prog-net` | ProgressiveNet | Progressive neural networks |
| `packnet` | PackNet | Pruning-based |
| `fame` | FAME | Fast Adaptive Meta-Emergence |

## Games

| Game | Modes |
|------|-------|
| `ALE/Freeway-v5` | 8 game difficulty variants |
| `ALE/SpaceInvaders-v5` | 10 game difficulty variants |

## Output

- `agents/` — Trained model checkpoints
- `runs/` — TensorBoard logs
- `eval_results_Freeway.csv` / `eval_results_SpaceInvaders.csv` — Evaluation results
- `data_FAME/envs/{Game}/task_{mode}.csv` — Per-task training curves
- `summary_data_{Game}_FAME.csv` — Final metrics
- `success_curves_{Game}_FAME.pdf` — Success rate plots
- `benchmark_*.log` — Full execution log
- `benchmark_errors.log` — Error log

## Resume

The script is resumable. If interrupted, re-running will skip already-completed training (checked by agent directory existence) and evaluation entries (checked by CSV content).

## Individual Runs

For single-algorithm runs:

```bash
# Baseline
python3 run_experiments.py --algorithm=from-scratch --env=ALE/Freeway-v5 --seed=1 --timesteps=1000000

# FAME (uses separate script)
python3 run_ppo_FAME.py --model-type=FAME --env-id=ALE/Freeway-v5 --seed=1 --save-dir=agents --total-timesteps=1000000 --epoch_meta=200
```

## Evaluation Only

```bash
./benchmark_experiments.sh --skip-training
```