# QR-DQN Terminal Logging Design

## Overview

Add compact one-line terminal logging to the QR-DQN training loop, printing training metrics every 10,000 frames.

## Format

```
frame=10000 | loss=0.42 | eps=0.82 | ep_return=2.3 | eval_return=- | fps=342 | elapsed=0:00:29
```

## Metrics

| Field | Source | Notes |
|---|---|---|
| frame | loop counter | Current frame count |
| loss | mean of losses since last log | Reset every 10k frames |
| eps | `get_epsilon(frame, config)` | Current exploration rate |
| ep_return | mean of episode returns since last log | `-` if no episodes ended in window |
| eval_return | last evaluation mean return | `-` if no evaluation has run yet |
| fps | frames / wall-clock seconds | Since last log point |
| elapsed | wall-clock since training start | `H:MM:SS` format |

## Implementation

### Config change (`qr_dqn/configs.py`)

Add `log_interval: int = 10000` to `QRDQNConfig`.

### Training loop changes (`qr_dqn/train.py`)

1. Track `start_time = time.time()` and `last_log_time = start_time` at loop start
2. Maintain `losses_since_log: list[float]` and `ep_returns_since_log: list[float]` accumulators
3. Append loss to `losses_since_log` after each training step
4. Append episode return to `ep_returns_since_log` on episode end
5. Every `log_interval` frames:
   - Compute mean loss, mean ep_return, fps, elapsed
   - Get current epsilon from `get_epsilon(frame, config)`
   - Print the compact one-liner
   - Reset accumulators
6. After each evaluation: store `last_eval_return` for display
7. Print final summary when training completes (total frames, final eval return, total elapsed)

### No new files or dependencies

All changes are inline in existing files. No new modules, classes, or external packages.

## Scope

Terminal logging only. No TensorBoard, WandB, file logging, or checkpointing in this change.