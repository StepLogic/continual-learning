# Atari Continual Learning Benchmark Script Design

**Date:** 2026-04-14  
**Paper:** Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning (ICLR 2026, https://openreview.net/forum?id=loNTDX3wTn)

## Goal

Produce a single `benchmark_experiments.sh` script that reproduces the FAME paper's Atari experiments end-to-end: training, evaluation, and results processing.

## Scope

- **2 games**: Freeway (8 modes), SpaceInvaders (10 modes)
- **6 algorithms**: cnn-simple, cnn-simple-ft (finetune), cnn-componet, prog-net, packnet, FAME
- **3 seeds**: 1, 2, 3
- **1M timesteps** per task
- **Full continual evaluation**: every trained agent tested on all game modes
- **Results processing**: TensorBoard CSV extraction + continual RL metrics (performance, forgetting, forward transfer)

## Architecture

Single bash script with 3 sequential phases.

### Phase 1: Training

Sequential execution: for each game → algorithm → seed → mode (0..N-1).

- **Training order matters**: modes must be trained sequentially (0, 1, 2, ...) because continual methods (finetune, componet, prog-net, packnet, FAME) depend on previously trained agents.
- **Script routing**: FAME uses `run_ppo_FAME.py`; all others use `run_experiments.py` (which internally calls `run_ppo.py`).
- **Resume capability**: skip training if the agent checkpoint directory already exists at `agents/ALE-{Game}-v5_{mode}__{algo}__run_ppo__{seed}/`.

Algorithm-specific CLI args:
- `cnn-componet`: `--componet-finetune-encoder`
- `packnet`: `--total-task-num={N_modes}`
- `dino-simple`: `--dino-size=s --num-envs=2`
- `cnn-simple-ft` / `packnet`: `--prev-units` pointing to the previous mode's agent
- `cnn-componet` / `prog-net`: `--prev-units` pointing to ALL previous mode agents
- FAME: `--buffer-path data_FAME/{game}_buffer_`

### Phase 2: Evaluation

For each game → algorithm → seed → training_mode → test_mode:
- Call `test_agent.py --load {agent_path} --mode {test_mode} --num-episodes 10 --max-timesteps 1000`
- Append results to `eval_results_{Game}.csv`
- Skip if result already exists in the CSV

Output CSV format: `algorithm,environment,train_mode,test_mode,seed,ep_ret`

### Phase 3: Results Processing

1. `process_results_pre.py` — extract TensorBoard event data → per-task CSV files in `data_FAME/envs/{Game}/`
2. `process_results.py` — compute success scores, forward transfer, forgetting, and final performance metrics
3. Generate summary tables and success curve plots

## Hyperparameters (from paper/code)

| Parameter | Value | Notes |
|-----------|-------|-------|
| total_timesteps | 1,000,000 | Per task |
| learning_rate | 2.5e-4 | All PPO-based |
| num_envs | 8 | (2 for dino-simple) |
| num_steps | 128 | Rollout length |
| update_epochs | 4 | PPO epochs |
| clip_coef | 0.1 | PPO clipping |
| FAME: lr_fast | 2.5e-4 | Fast learner LR |
| FAME: lr_meta | 2.5e-4 | Meta learner LR |
| FAME: size_fast2meta | 20,000 | Fast→meta buffer size |
| FAME: size_meta | 200,000 | Meta replay buffer |
| FAME: detection_step | 1,200 | Policy eval steps |
| FAME: epoch_meta | 200 | Meta training epochs |
| FAME: warmstep | 50,000 | BC warmup steps |
| FAME: lambda_reg | 1.0 | BC regularization weight |
| PackNet: total_task_num | 8 (Freeway) / 10 (SpaceInvaders) | Total tasks for mask allocation |

## Error Handling

- `set -e` is NOT used — failed runs log the error and continue
- Each failed training run is logged to `benchmark_errors.log`
- Final summary prints pass/fail counts per algorithm per game
- Exit code is 0 only if all runs succeeded

## Output Files

| Path | Description |
|------|-------------|
| `agents/` | Trained model checkpoints |
| `runs/` | TensorBoard logs |
| `eval_results_Freeway.csv` | Freeway evaluation CSV |
| `eval_results_SpaceInvaders.csv` | SpaceInvaders evaluation CSV |
| `data_FAME/envs/{Game}/task_{mode}.csv` | Per-task training curves |
| `summary_data_{Game}_FAME.csv` | Final metrics table |
| `success_curves_{Game}_FAME.pdf` | Success rate plots |
| `benchmark_errors.log` | Error log |

## Estimated Runtime

- Freeway: 8 modes × 6 algos × 3 seeds = 144 task-runs × ~25 min = ~60 hours
- SpaceInvaders: 10 modes × 6 algos × 3 seeds = 180 task-runs × ~25 min = ~75 hours
- Evaluation: ~30 min per game
- **Total: ~135 hours** (single GPU, serial)

## CLI Interface

```bash
./benchmark_experiments.sh [OPTIONS]

Options:
  --env=ENV          Game environment (default: both Freeway and SpaceInvaders)
  --seed=SEEDS       Comma-separated seeds (default: 1,2,3)
  --timesteps=N      Timesteps per task (default: 1000000)
  --algorithms=LIST  Comma-separated algorithm list (default: all)
  --skip-training    Skip to evaluation phase only
  --skip-eval        Skip evaluation, only train
  --quick            Quick test: 10000 timesteps, 1 seed, 1 game
  -h, --help         Show help
```