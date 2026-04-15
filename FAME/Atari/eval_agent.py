"""
eval_agent.py  —  Deterministic evaluation of a saved checkpoint on one task.

Called by run_atari.py after each training phase.  Writes a small JSON with:
    { mean_return, std_return, random_return, episodes, task_id, checkpoint }

Usage (examples):
    python eval_agent.py --model-type=FAME          --env-id=ALE/Freeway-v5 \
        --mode=2 --checkpoint=agents/ALE-Freeway-v5_2__FAME__run_ppo__1 \
        --eval-episodes=5 --output=eval_logs/eval_train2_on_task2_FAME_1.json

    python eval_agent.py --model-type=packnet       --env-id=ALE/Freeway-v5 \
        --mode=1 --checkpoint=agents/...  --task-id=2 --eval-episodes=5

    python eval_agent.py --model-type=cnn-componet  --env-id=ALE/Freeway-v5 \
        --mode=0 --checkpoint=agents/...  --eval-episodes=5 \
        --prev-units agents/task0_ckpt agents/task1_ckpt
"""

import argparse
import json
import os
import pathlib
import random
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import ale_py

gym.register_envs(ale_py)

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv,
)
from models import (
    CnnSimpleAgent, DinoSimpleAgent, CnnCompoNetAgent,
    ProgressiveNetAgent, PackNetAgent, FAMEAgent,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",     required=True)
    p.add_argument("--env-id",         required=True)
    p.add_argument("--mode",           type=int, required=True)
    p.add_argument("--checkpoint",     required=True,
                   help="Path to the saved agent directory")
    p.add_argument("--eval-episodes",  type=int, default=5)
    p.add_argument("--seed",           type=int, default=1)
    p.add_argument("--output",         type=str, default=None,
                   help="JSON file to write results to")

    # algorithm-specific extras
    p.add_argument("--prev-units", nargs="*", default=[],
                   help="Previous checkpoint dirs (componet / prog-net)")
    p.add_argument("--task-id",    type=int,  default=None,
                   help="1-indexed task id for PackNet mask selection")
    p.add_argument("--dino-size",  type=str,  default="s")
    return p.parse_args()


def make_eval_env(env_id: str, mode: int, seed: int, dino: bool = False):
    """Single deterministic evaluation environment (no video capture)."""
    env = gym.make(env_id, mode=mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    if not dino:
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
    else:
        env = gym.wrappers.ResizeObservation(env, (224, 224))
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


def random_policy_return(env, episodes: int = 3, seed: int = 0) -> float:
    """Baseline: pure random policy average return."""
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            ep_return += reward
            done = term or trunc
        returns.append(ep_return)
    return float(np.mean(returns))


def load_agent(args, env, device):
    """Load the correct agent class from the checkpoint directory."""
    m = args.model_type
    ckpt = args.checkpoint

    if m == "cnn-simple" or m == "cnn-simple-ft":
        agent = CnnSimpleAgent.load(ckpt, env).to(device)

    elif m == "dino-simple":
        agent = DinoSimpleAgent.load(
            ckpt, env, dino_size=args.dino_size, frame_stack=4, device=device
        ).to(device)

    elif m == "cnn-componet":
        # prev-units contains ALL units including the current one
        agent = CnnCompoNetAgent.load(
            ckpt,
            prevs_paths=[pathlib.Path(p) for p in args.prev_units[:-1]],
            map_location=device,
        ).to(device)

    elif m == "prog-net":
        agent = ProgressiveNetAgent.load(
            ckpt,
            prevs_paths=[pathlib.Path(p) for p in args.prev_units[:-1]],
            map_location=device,
        ).to(device)

    elif m == "packnet":
        task_id = args.task_id if args.task_id is not None else (args.mode + 1)
        agent = PackNetAgent.load(
            ckpt, task_id=task_id, restart_actor_critic=False, freeze_bias=True
        ).to(device)
        agent.network.set_view(task_id)   # activate the correct binary mask

    elif m == "FAME":
        # Evaluate the fast-learner checkpoint for the given task
        agent = FAMEAgent.load(ckpt, env, fast=True).to(device)

    else:
        raise ValueError(f"Unknown model type: {m}")

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(agent, env, episodes: int, device, seed: int = 0) -> List[float]:
    """
    Run `episodes` deterministic episodes.
    Determinism: we pass --deterministic=True at the action-selection level
    (argmax rather than sampling) where the agent supports it; otherwise we
    sample normally (the stochasticity is acceptable for evaluation).
    """
    agent.eval()
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        ep_return = 0.0

        while not done:
            # Use greedy action selection (argmax of actor logits) for eval
            action, _, _, _ = agent.get_action_and_value(obs / 255.0,
                                                          deterministic=True)

            action_np = action.cpu().numpy().flatten()
            next_obs, reward, term, trunc, info = env.step(action_np[0])
            ep_return += reward
            done = term or trunc
            obs = torch.tensor(next_obs, dtype=torch.float32,
                               device=device).unsqueeze(0)

        returns.append(ep_return)
        print(f"    Episode {ep+1}/{episodes}  return={ep_return:.1f}")

    return returns


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dino = "dino" in args.model_type
    env  = make_eval_env(args.env_id, args.mode, args.seed, dino=dino)

    # ── Random baseline (cheap: 3 episodes) ───────────────────────────────────
    rand_return = random_policy_return(env, episodes=3, seed=args.seed)
    print(f"  Random baseline on task {args.mode}: {rand_return:.2f}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    # Build a tiny vectorised env just for shape inference (some loaders need it)
    vec_env = gym.vector.SyncVectorEnv(
        [lambda: make_eval_env(args.env_id, args.mode, args.seed, dino=dino)]
    )
    agent = load_agent(args, vec_env, device)
    vec_env.close()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n  Evaluating checkpoint '{args.checkpoint}' on task {args.mode} "
          f"({args.eval_episodes} episodes) ...")
    episode_returns = evaluate(agent, env, args.eval_episodes, device, args.seed)
    env.close()

    mean_r = float(np.mean(episode_returns))
    std_r  = float(np.std(episode_returns))
    print(f"\n  Result: mean={mean_r:.2f}  std={std_r:.2f}  "
          f"random_baseline={rand_return:.2f}")

    result = {
        "mean_return":    mean_r,
        "std_return":     std_r,
        "random_return":  rand_return,
        "episodes":       episode_returns,
        "eval_episodes":  args.eval_episodes,
        "checkpoint":     args.checkpoint,
        "task_id":        args.mode,
        "model_type":     args.model_type,
        "seed":           args.seed,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()