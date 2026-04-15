"""
run_atari.py  —  Sequential task training + full continual learning evaluation.

After each task k is trained, every previously saved agent checkpoint is evaluated
on tasks 0 … k.  This gives the full R[i,j] matrix needed to compute:

    • Average Performance      AP   = mean of final row
    • Forgetting               F    = mean drop from peak on old tasks
    • Forward Transfer         FWT  = how much knowing task k-1 helps task k
    • Backward Transfer        BWT  = average change on old tasks after more training

Usage (same as before):
    python run_atari.py --algorithm fame  --env ALE/Freeway-v5   --seed 1 --timesteps 1000000
    python run_atari.py --algorithm packnet --env ALE/SpaceInvaders-v5
    python run_atari.py --algorithm prog-net --env ALE/Breakout-v5 --start-mode 0
"""

import subprocess
import argparse
import random
import os
import time
import json
import numpy as np
from task_utils import TASKS

# Resolve sibling script paths relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_PPO_SCRIPT = os.path.join(SCRIPT_DIR, "run_ppo.py")
RUN_FAME_SCRIPT = os.path.join(SCRIPT_DIR, "run_ppo_FAME.py")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_agent.py")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str,
                        choices=["cnn-simple", "componet", "finetune",
                                 "from-scratch", "prog-net", "packnet",
                                 "fame", "dino-simple"],
                        default="fame")
    parser.add_argument("--env", type=str,
                        choices=["ALE/Breakout-v5",
                                 "ALE/SpaceInvaders-v5",
                                 "ALE/Freeway-v5"],
                        default="ALE/Freeway-v5")
    parser.add_argument("--seed", type=int, default=1, help="[1,10]")
    parser.add_argument("--start-mode", type=int, default=0)
    parser.add_argument("--dino-size", type=str,
                        choices=["s", "b", "l", "g"], default="s")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Timesteps per task (use 1e6 for full runs)")
    # ── evaluation knobs ──────────────────────────────────────────────────────
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Deterministic evaluation episodes per (agent, task) pair")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip backward evaluation (training only, faster)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip training/evaluation if checkpoints/logs already exist")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ALGO_TO_MODEL = {
    "cnn-simple":   "cnn-simple",
    "finetune":     "cnn-simple-ft",
    "componet":     "cnn-componet",
    "from-scratch": "cnn-simple",
    "dino-simple":  "dino-simple",
    "prog-net":     "prog-net",
    "packnet":      "packnet",
    "fame":         "FAME",
}


def run_name_fn(env: str, model_type: str, seed: int, algorithm: str = None):
    """Returns a callable: task_id → checkpoint directory name.

    FAME uses 'run_ppo_FAME' as the experiment name suffix because
    run_ppo_FAME.py derives it from the filename.  All other methods
    use 'run_ppo' (from run_ppo.py).
    """
    if algorithm == "fame":
        exp_name = "run_ppo_FAME"
    else:
        exp_name = "run_ppo"
    return lambda task_id: (
        f"{env.replace('/', '-')}_{task_id}__{model_type}__{exp_name}__{seed}"
    )


def build_train_cmd(args, model_type, seed, task_id, modes, run_name, timesteps):
    """Build the training subprocess command for one task (per-task methods only)."""
    script = RUN_PPO_SCRIPT

    params = (
        f"--model-type={model_type} --env-id={args.env} --seed={seed}"
        f" --mode={task_id} --save-dir=agents --total-timesteps={timesteps}"
    )

    if args.algorithm == "componet":
        params += " --componet-finetune-encoder"
    if args.algorithm == "packnet":
        params += f" --total-task-num={len(modes)}"
    if args.algorithm == "dino-simple":
        params += f" --dino-size={args.dino_size} --num-envs=1"

    # Pass previous checkpoints where required
    task_idx = modes.index(task_id)
    if task_idx > 0:
        if args.algorithm in ["componet", "prog-net"]:
            params += " --prev-units"
            for prev in modes[:task_idx]:
                params += f" agents/{run_name(prev)}"
        elif args.algorithm in ["finetune", "packnet"]:
            params += f" --prev-units agents/{run_name(modes[task_idx - 1])}"

    return f"python3 {script} {params}"


def build_fame_train_cmd(args, model_type, seed, timesteps):
    """Build the training command for FAME (trains ALL modes in a single call).

    FAME's run_ppo_FAME.py internally iterates over all modes in TASKS[env_id],
    so we call it once per (env, seed) rather than once per task.
    """
    game = args.env.split("/")[-1].replace("-v5", "")
    params = (
        f"--model-type={model_type} --env-id={args.env} --seed={seed}"
        f" --save-dir=agents --total-timesteps={timesteps}"
        f" --buffer_path data_FAME/{game}_buffer_"
    )
    return f"python3 {RUN_FAME_SCRIPT} {params}"


def build_eval_cmd(args, model_type, seed, train_task_id, eval_task_id,
                   modes, run_name, eval_episodes):
    """
    Build an evaluation command.

    The eval script (eval_agent.py) loads the checkpoint trained on
    `train_task_id` and evaluates it on `eval_task_id` deterministically.

    For algorithms that carry ALL previous columns (componet, prog-net) we pass
    the full prev-units list so the agent can reconstruct itself correctly.
    """
    checkpoint = f"agents/{run_name(train_task_id)}"
    params = (
        f"--model-type={model_type} --env-id={args.env} --seed={seed}"
        f" --mode={eval_task_id} --checkpoint={checkpoint}"
        f" --eval-episodes={eval_episodes}"
    )

    if args.algorithm in ["componet", "prog-net"]:
        train_idx = modes.index(train_task_id)
        params += " --prev-units"
        for prev in modes[:train_idx]:          # all units trained before train_task
            params += f" agents/{run_name(prev)}"
        params += f" agents/{run_name(train_task_id)}"  # the unit itself

    if args.algorithm == "packnet":
        params += f" --task-id={eval_task_id + 1}"     # PackNet uses 1-indexed task IDs

    # FAME evaluation uses the fast-learner checkpoint transparently via --checkpoint

    return f"python3 {EVAL_SCRIPT} {params}"


# ─────────────────────────────────────────────────────────────────────────────
# CL Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_cl_metrics(R: np.ndarray, R_random: np.ndarray) -> dict:
    """
    R[i, j] = performance of agent trained up-to task i, evaluated on task j
              (only defined for j <= i, upper triangle is np.nan)

    R_random[j] = random-policy baseline on task j (used for normalisation)

    Returns
    -------
    dict with keys: AP, Forgetting, BWT, FWT, R_matrix (list-of-lists)
    """
    N = R.shape[0]

    # ── Average Performance (final agent on all tasks) ─────────────────────
    AP = float(np.nanmean(R[N - 1, :]))          # last row

    # ── Forgetting ─────────────────────────────────────────────────────────
    # For each task j < N-1: max performance seen so far minus final performance
    forgetting_per_task = []
    for j in range(N - 1):
        col = R[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            forgetting_per_task.append(float(np.max(valid) - valid[-1]))
    Forgetting = float(np.mean(forgetting_per_task)) if forgetting_per_task else 0.0

    # ── Backward Transfer ──────────────────────────────────────────────────
    # BWT[j] = R[N-1, j] - R[j, j]  (did later training help/hurt old tasks?)
    bwt_per_task = []
    for j in range(N - 1):
        if not np.isnan(R[N - 1, j]) and not np.isnan(R[j, j]):
            bwt_per_task.append(float(R[N - 1, j] - R[j, j]))
    BWT = float(np.mean(bwt_per_task)) if bwt_per_task else 0.0

    # ── Forward Transfer ───────────────────────────────────────────────────
    # FWT[j] = R[j-1, j] - R_random[j]  (does prior knowledge help on task j?)
    fwt_per_task = []
    for j in range(1, N):
        if not np.isnan(R[j - 1, j]):
            fwt_per_task.append(float(R[j - 1, j] - R_random[j]))
    FWT = float(np.mean(fwt_per_task)) if fwt_per_task else float("nan")

    return {
        "AP":         AP,
        "Forgetting": Forgetting,
        "BWT":        BWT,
        "FWT":        FWT,
        "R_matrix":   np.where(np.isnan(R), None, R).tolist(),
    }


def print_metrics(metrics: dict, modes: list):
    N = len(modes)
    print("\n" + "=" * 60)
    print("  CONTINUAL LEARNING EVALUATION RESULTS")
    print("=" * 60)

    # Print R matrix
    print(f"\n  Performance Matrix  R[train_task, eval_task]")
    header = "        " + "".join(f"  T{m:<3}" for m in modes)
    print(header)
    for i, row in enumerate(metrics["R_matrix"]):
        vals = "".join(
            f"  {v:5.1f}" if v is not None else "     —" for v in row
        )
        print(f"  T{modes[i]:<5}{vals}")

    print(f"\n  Average Performance (AP) : {metrics['AP']:8.3f}")
    print(f"  Forgetting               : {metrics['Forgetting']:8.3f}  (lower is better)")
    print(f"  Backward Transfer (BWT)  : {metrics['BWT']:8.3f}  (higher is better)")
    print(f"  Forward Transfer  (FWT)  : {metrics['FWT']:8.3f}  (higher is better)")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_cmd(cmd, log_path, skip_existing=False):
    """Run an evaluation command and return the mean_return, or None on failure."""
    if skip_existing and os.path.exists(log_path):
        try:
            with open(log_path) as f:
                result = json.load(f)
            mean_return = result.get("mean_return", float("nan"))
            print(f"      (cached) mean_return = {mean_return:.2f}")
            return mean_return
        except (json.JSONDecodeError, OSError):
            pass  # re-run if log is corrupt

    res = subprocess.run(cmd, shell=True)
    if res.returncode == 0 and os.path.exists(log_path):
        try:
            with open(log_path) as f:
                result = json.load(f)
            mean_return = result.get("mean_return", float("nan"))
            print(f"      → mean_return = {mean_return:.2f}")
            return mean_return
        except (json.JSONDecodeError, OSError):
            print(f"      *** Eval log corrupt — storing NaN")
            return float("nan")
    else:
        print(f"      *** Eval failed or log missing — storing NaN")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    modes      = TASKS[args.env]
    seed       = args.seed
    model_type = ALGO_TO_MODEL[args.algorithm]
    run_name   = run_name_fn(args.env, model_type, seed, algorithm=args.algorithm)
    timesteps  = args.timesteps

    first_idx  = modes.index(args.start_mode)
    task_seq   = modes[first_idx:]          # tasks we will actually train
    N          = len(modes)                 # total number of tasks

    # R[i, j] — performance of checkpoint i on task j; NaN = not yet evaluated
    R        = np.full((N, N), np.nan)
    R_random = np.full(N, np.nan)           # populated during first evaluation pass

    results_path = f"cl_results_{args.env.replace('/', '-')}_{model_type}_{seed}.json"
    os.makedirs("agents",   exist_ok=True)
    os.makedirs("eval_logs", exist_ok=True)

    # ── FAME: single training call for all modes ───────────────────────────
    if args.algorithm == "fame":
        print(f"\n{'='*60}")
        print(f"  TRAINING  FAME on {args.env}  (all modes at once)",
              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"{'='*60}")

        # Check if all FAME checkpoints already exist
        all_exist = all(
            os.path.exists(f"agents/{run_name(m)}") for m in modes
        )
        if all_exist and args.skip_existing:
            print("  All FAME checkpoints exist, skipping training.")
        else:
            train_cmd = build_fame_train_cmd(args, model_type, seed, timesteps)
            print(f"  CMD: {train_cmd}\n")
            res = subprocess.run(train_cmd, shell=True)
            if res.returncode != 0:
                print(f"  *** FAME training failed (code {res.returncode}).")
                return

        if args.skip_eval:
            print("\nTraining complete. Evaluation skipped (--skip-eval).")
            return

        # ── Evaluate all FAME checkpoints on all tasks ────────────────────
        for i, train_task_id in enumerate(modes):
            ckpt = f"agents/{run_name(train_task_id)}"
            if not os.path.exists(ckpt):
                print(f"  Checkpoint {ckpt} not found, skipping evaluation.")
                continue

            global_task_idx = modes.index(train_task_id)

            # Backward evaluation: checkpoint trained up to train_task_id on tasks 0..train_task_id
            print(f"\n  --- Backward eval: checkpoint T{train_task_id} on tasks 0..T{train_task_id} ---")
            for eval_task_id in modes[: global_task_idx + 1]:
                eval_j = modes.index(eval_task_id)

                eval_cmd = build_eval_cmd(
                    args, model_type, seed,
                    train_task_id=train_task_id,
                    eval_task_id=eval_task_id,
                    modes=modes,
                    run_name=run_name,
                    eval_episodes=args.eval_episodes,
                )
                print(f"    Eval checkpoint T{train_task_id} → env T{eval_task_id}")
                print(f"    CMD: {eval_cmd}")

                log_path = (
                    f"eval_logs/eval_train{train_task_id}_on_task{eval_task_id}"
                    f"_{model_type}_{seed}.json"
                )
                eval_cmd_logged = eval_cmd + f" --output={log_path}"
                mean_return = run_eval_cmd(eval_cmd_logged, log_path, args.skip_existing)
                R[global_task_idx, eval_j] = mean_return

                # Populate random baseline on first encounter
                if not np.isnan(mean_return) and os.path.exists(log_path):
                    with open(log_path) as f:
                        result = json.load(f)
                    rand_r = result.get("random_return", 0.0)
                    if np.isnan(R_random[eval_j]):
                        R_random[eval_j] = rand_r

            # Forward evaluation: previous checkpoints on the new task
            if global_task_idx > 0:
                print(f"\n  --- Forward eval: old checkpoints on task T{train_task_id} ---")
                for prev_task_id in modes[:global_task_idx]:
                    prev_i = modes.index(prev_task_id)
                    prev_ckpt = f"agents/{run_name(prev_task_id)}"
                    if not os.path.exists(prev_ckpt):
                        continue

                    eval_cmd = build_eval_cmd(
                        args, model_type, seed,
                        train_task_id=prev_task_id,
                        eval_task_id=train_task_id,
                        modes=modes,
                        run_name=run_name,
                        eval_episodes=args.eval_episodes,
                    )
                    log_path = (
                        f"eval_logs/eval_train{prev_task_id}_on_task{train_task_id}"
                        f"_{model_type}_{seed}.json"
                    )
                    eval_cmd_logged = eval_cmd + f" --output={log_path}"
                    print(f"    Eval checkpoint T{prev_task_id} → env T{train_task_id}")
                    mean_return = run_eval_cmd(eval_cmd_logged, log_path, args.skip_existing)
                    R[prev_i, global_task_idx] = mean_return

        # Final report
        metrics = compute_cl_metrics(R, R_random)
        with open(results_path, "w") as f:
            json.dump({"R": R.tolist(), "R_random": R_random.tolist(),
                       "metrics": metrics, "modes": modes,
                       "algorithm": args.algorithm, "env": args.env,
                       "seed": seed}, f, indent=2)
        print_metrics(metrics, modes)
        print(f"Final results saved → {results_path}")

    # ── Per-task methods: train one task at a time ─────────────────────────
    else:
        for i, task_id in enumerate(task_seq):
            global_task_idx = modes.index(task_id)   # index in the full mode list

            # ── 1. TRAIN ──────────────────────────────────────────────────────
            print(f"\n{'='*60}")
            print(f"  TRAINING  task {task_id}  ({i+1}/{len(task_seq)})",
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(f"{'='*60}")

            ckpt_dir = f"agents/{run_name(task_id)}"
            if args.skip_existing and os.path.exists(ckpt_dir):
                print(f"  Checkpoint {ckpt_dir} exists, skipping training.")
            else:
                train_cmd = build_train_cmd(
                    args, model_type, seed, task_id, modes, run_name, timesteps
                )
                print(f"  CMD: {train_cmd}\n")
                res = subprocess.run(train_cmd, shell=True)
                if res.returncode != 0:
                    print(f"  *** Training task {task_id} failed (code {res.returncode}). Skipping eval.")
                    continue

            if args.skip_eval:
                continue

            # ── 2. EVALUATE current checkpoint on ALL tasks trained so far ────
            print(f"\n  --- Backward evaluation after training task {task_id} ---")
            for eval_task_id in modes[: global_task_idx + 1]:
                eval_j = modes.index(eval_task_id)

                eval_cmd = build_eval_cmd(
                    args, model_type, seed,
                    train_task_id=task_id,
                    eval_task_id=eval_task_id,
                    modes=modes,
                    run_name=run_name,
                    eval_episodes=args.eval_episodes,
                )
                print(f"    Eval checkpoint T{task_id} → env T{eval_task_id}")
                print(f"    CMD: {eval_cmd}")

                log_path = (
                    f"eval_logs/eval_train{task_id}_on_task{eval_task_id}"
                    f"_{model_type}_{seed}.json"
                )
                eval_cmd_logged = eval_cmd + f" --output={log_path}"
                mean_return = run_eval_cmd(eval_cmd_logged, log_path, args.skip_existing)
                R[global_task_idx, eval_j] = mean_return

                # Populate random baseline on first encounter
                if not np.isnan(mean_return) and os.path.exists(log_path):
                    with open(log_path) as f:
                        result = json.load(f)
                    rand_r = result.get("random_return", 0.0)
                    if np.isnan(R_random[eval_j]):
                        R_random[eval_j] = rand_r

            # ── 3. Forward evaluation: previous checkpoints on the NEW task ──
            if global_task_idx > 0:
                print(f"\n  --- Forward evaluation: old checkpoints on new task T{task_id} ---")
                for prev_task_id in modes[:global_task_idx]:
                    prev_i = modes.index(prev_task_id)

                    # Skip if the checkpoint doesn't exist yet (start_mode > 0 case)
                    ckpt = f"agents/{run_name(prev_task_id)}"
                    if not os.path.exists(ckpt):
                        continue

                    eval_cmd = build_eval_cmd(
                        args, model_type, seed,
                        train_task_id=prev_task_id,
                        eval_task_id=task_id,
                        modes=modes,
                        run_name=run_name,
                        eval_episodes=args.eval_episodes,
                    )
                    log_path = (
                        f"eval_logs/eval_train{prev_task_id}_on_task{task_id}"
                        f"_{model_type}_{seed}.json"
                    )
                    eval_cmd_logged = eval_cmd + f" --output={log_path}"
                    print(f"    Eval checkpoint T{prev_task_id} → env T{task_id}")
                    mean_return = run_eval_cmd(eval_cmd_logged, log_path, args.skip_existing)
                    R[prev_i, global_task_idx] = mean_return

            # ── 4. Snapshot intermediate metrics ──────────────────────────────
            metrics = compute_cl_metrics(R, R_random)
            with open(results_path, "w") as f:
                json.dump({"R": R.tolist(), "R_random": R_random.tolist(),
                           "metrics": metrics, "modes": modes,
                           "algorithm": args.algorithm, "env": args.env,
                           "seed": seed}, f, indent=2)
            print(f"\n  Intermediate results saved → {results_path}")

        # ── Final report ──────────────────────────────────────────────────────
        if not args.skip_eval:
            metrics = compute_cl_metrics(R, R_random)
            print_metrics(metrics, modes)
            print(f"Final results saved → {results_path}")
        else:
            print("\nTraining complete. Evaluation skipped (--skip-eval).")


if __name__ == "__main__":
    main()