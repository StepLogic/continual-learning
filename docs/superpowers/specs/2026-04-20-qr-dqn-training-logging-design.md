# QR-DQN Training Info Logging — Design Spec

**Date:** 2026-04-20
**Status:** Approved
**Purpose:** Add structured terminal logging for QR-DQN training metrics.

## Overview

Add frame-interval-based terminal logging to `train.py` that prints key training metrics at configurable intervals. No external logging backends — just `print` statements. Pattern borrowed from jaxrl5's approach of keeping logging simple and in the training loop.

## Changes

### `configs.py` — Add `log_interval`

```python
log_interval: int = 10_000  # frames between log prints
```

### `train.py` — Add `_log_metrics()` helper + wire into loop

A private helper function that:
- Takes current frame, averaged loss since last log, epsilon, episode count, recent returns, max return
- Prints one formatted line per interval

**Loss accumulation:** Collect losses in a list between log intervals. At each `log_interval` boundary, compute mean loss and print. Reset the accumulator.

**Printed format:**
```
[Frame 10000] loss=0.523 | eps=0.94 | episodes=42 | mean_ret(10)=1.2 | max_ret=3.0
[Frame 20000] loss=0.341 | eps=0.88 | episodes=85 | mean_ret(10)=2.8 | max_ret=5.0
Eval @ 25000: mean_return=4.2 over 10 episodes
```

### No new files, no new classes, no external dependencies.

## What Stays the Same

- `agent.train_step()` returns `{"loss": float}` — no agent changes
- Episode-end `logger.info` calls unchanged
- Final JSON dump in `run_qr_dqn.py` unchanged

## Metrics Logged

| Metric | Source | When |
|--------|--------|------|
| loss (avg over interval) | `train_step()` return | Every `log_interval` frames |
| epsilon | `get_epsilon()` | Every `log_interval` frames |
| episode count | counter in loop | Every `log_interval` frames |
| mean return (last 10) | `episode_returns` list | Every `log_interval` frames |
| max return | tracked in loop | Every `log_interval` frames |
| eval mean return | `evaluate()` return | Every `eval_interval` frames |

## Testing

- Existing tests continue to pass (no API changes to agent or config defaults)
- Manual smoke test: run training briefly, verify printed output format