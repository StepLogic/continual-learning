# Continual RL Benchmark

Benchmark suite for comparing SOTA Continual Reinforcement Learning methods.

## Methods Implemented

| Method | Venue | Description |
|--------|-------|-------------|
| Fast TRAC | NeurIPS 2024 | Parameter-free optimizer for plasticity |
| Parseval Regularization | NeurIPS 2024 | Orthogonality-constrained optimization |
| Dual-Learner | ICLR 2026 | Fast + Meta knowledge learners |
| Online World Model | ICML 2025 | Planning with learned dynamics |
| Dual Actor | ICLR 2025 | Reset + distillation policies |

## Installation

```bash
pip install -e .
pip install -e ".[envs]"  # For environments
pip install -e ".[dev]"   # For development
```

## Quick Start

```bash
# Run a single method on Continual World
python experiments/run_all.py method=fast_trac env=continual_world seed=0

# Run all methods (smoke test)
python experiments/run_all.py --smoke-test

# Full benchmark
python experiments/run_all.py --full-benchmark
```

## Results

Results are saved to `results/`:
- `tensorboard/` - Training curves
- `csv/` - Raw metrics per run
- `reports/` - Generated summaries

## Project Structure

```
continual_rl_benchmark/
├── configs/          # Hydra configurations
├── src/
│   ├── agents/       # RL algorithms and methods
│   ├── environments/ # Continual RL environments
│   ├── evaluation/   # Metrics and evaluators
│   └── utils/        # Logging, helpers
├── experiments/      # Experiment runners
├── scripts/          # Analysis and plotting
├── results/          # Output data
└── tests/            # Unit tests
```

## Citing

If you use this benchmark, please cite the original papers.
