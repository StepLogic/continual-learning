"""
ContinualBench Benchmark Runner.

Integrates ContinualBench environments with our continual RL agents.
ContinualBench provides unified world dynamics for continual RL evaluation.

Installation:
    git clone https://github.com/sail-sg/ContinualBench.git
    cd ContinualBench && pip install -e .

Usage:
    python experiments/run_continualbench.py --method all --seeds 0 1 2
    python experiments/run_continualbench.py --method base --env continual_bench
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

# Set headless rendering for HPC
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

try:
    from continual_bench.envs import ContinualBenchEnv
    CONTINUALBENCH_AVAILABLE = True
except ImportError:
    CONTINUALBENCH_AVAILABLE = False
    print("Warning: ContinualBench not installed. Install with:")
    print("  git clone https://github.com/sail-sg/ContinualBench.git")
    print("  cd ContinualBench && pip install -e .")

from agents import (
    SACAgent,
    FastTRACAgent,
    ParsevalAgent,
    DualLearnerAgent,
    WorldModelAgent,
    DualActorAgent,
)
from evaluation import ContinualMetrics, Evaluator
from utils.logging import setup_logging, get_logger


# Agent registry
AGENT_REGISTRY = {
    "base": SACAgent,
    "fast_trac": FastTRACAgent,
    "parseval": ParsevalAgent,
    "dual_learner": DualLearnerAgent,
    "world_model": WorldModelAgent,
    "dual_actor": DualActorAgent,
}


def get_agent_config(method_name: str) -> dict:
    """Get default configuration for a method."""
    configs = {
        "base": {
            "hidden_dim": 256,
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        },
        "fast_trac": {
            "hidden_dim": 256,
            "base_lr": 0.003,
            "momentum_decay": 0.99,
            "grad_scale_threshold": 1.0,
            "buffer_size": 100000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        },
        "parseval": {
            "hidden_dim": 256,
            "learning_rate": 3e-4,
            "parseval_lambda": 0.1,
            "orthogonality_freq": 10,
            "buffer_size": 100000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        },
        "dual_learner": {
            "hidden_dim": 256,
            "learning_rate": 3e-4,
            "fast_lr": 0.001,
            "meta_lr": 0.0001,
            "switch_threshold": 0.5,
            "buffer_size": 100000,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        },
        "world_model": {
            "hidden_dim": 256,
            "learning_rate": 3e-4,
            "model_hidden_dim": 512,
            "model_lr": 0.0003,
            "mpc_horizon": 10,
            "mpc_samples": 100,
            "model_update_freq": 10,
            "buffer_size": 100000,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        },
        "dual_actor": {
            "hidden_dim": 256,
            "learning_rate": 3e-4,
            "reset_interval": 50000,
            "distill_alpha": 0.5,
            "bc_buffer_size": 10000,
            "buffer_size": 100000,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        },
    }
    return configs.get(method_name, configs["base"])


def evaluate(
    agent,
    env,
    num_episodes: int = 10,
    deterministic: bool = True,
    success_threshold: float = 0.9,
) -> Tuple[float, float, float]:
    """
    Evaluate agent on current task.

    Returns average return, length, and success rate across episodes.
    Success is defined as achieving >= success_threshold of max possible reward.
    """
    returns = []
    lengths = []
    successes = []

    # Estimate max reward from environment or use heuristic
    max_reward_estimate = 10.0  # Adjust based on environment

    for _ in range(num_episodes):
        obs = env.reset()
        episode_return = 0
        episode_length = 0

        while True:
            action = agent.get_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            if done:
                break

        returns.append(episode_return)
        lengths.append(episode_length)
        # Success: episode reached threshold of max reward
        successes.append(1 if episode_return >= success_threshold * max_reward_estimate else 0)

    return np.mean(returns), np.mean(lengths), np.mean(successes)


def run_continualbench_experiment(
    method_name: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    steps_per_task: int = 100000,
    checkpoint_dir: str = "checkpoints",
    save_checkpoints: bool = True,
    render: bool = False,
) -> dict:
    """
    Run ContinualBench experiment.

    Args:
        method_name: Name of the method to run
        seed: Random seed
        total_steps: Total training steps
        eval_interval: Steps between evaluations
        eval_episodes: Episodes per evaluation
        steps_per_task: Steps per task
        checkpoint_dir: Directory to save checkpoints
        save_checkpoints: Whether to save checkpoints
        render: Whether to render environment

    Returns:
        Final metrics dictionary
    """
    if not CONTINUALBENCH_AVAILABLE:
        raise ImportError("ContinualBench not installed. Install with: pip install -e .")

    logger = get_logger()
    logger.info(f"Starting ContinualBench experiment: {method_name} (seed={seed})")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create ContinualBench environment
    env = ContinualBenchEnv(render_mode="human" if render else "rgb_array", seed=seed)

    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    logger.info(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    # Create agent
    config = get_agent_config(method_name)
    AgentClass = AGENT_REGISTRY.get(method_name, SACAgent)
    agent = AgentClass(obs_dim=obs_dim, act_dim=act_dim, **config)

    # Create evaluator
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{method_name}_continualbench_{timestamp}"
    evaluator = Evaluator(experiment_name=experiment_name, seed=seed)

    # Create metrics tracker
    # ContinualBench uses unified world dynamics with task switching
    num_tasks = 10  # Default number of tasks in ContinualBench
    metrics_tracker = ContinualMetrics(num_tasks=num_tasks)

    # Training loop
    obs = env.reset()
    episode_return = 0
    episode_length = 0
    task_idx = 0

    pbar = tqdm(range(total_steps), desc=f"Training {method_name} on ContinualBench")

    warmup_steps = min(10000, total_steps // 100)

    for step in pbar:
        # Get action
        if step < warmup_steps:
            action = agent.get_action(obs, random=True)
        else:
            action = agent.get_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        done = terminated or truncated

        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)

        # Update agent
        update_metrics = agent.update()

        # Track episode
        episode_return += reward
        episode_length += 1

        if done:
            # Log episode
            evaluator.log_episode(
                episode_return=episode_return,
                episode_length=episode_length,
                task_idx=task_idx,
                task_name=f"task_{task_idx}",
                task_switched=False,
                additional_metrics=update_metrics if update_metrics else None,
            )

            metrics_tracker.log_episode(episode_return, task_idx, evaluator.global_step)

            # Reset after episode
            obs = env.reset()
            episode_return = 0
            episode_length = 0
        else:
            obs = next_obs

        # Log training metrics
        if update_metrics:
            evaluator.log_training_metrics(update_metrics)

        # Periodic evaluation
        if step > 0 and step % eval_interval == 0:
            eval_return, eval_length, eval_success_rate = evaluate(
                agent, env, num_episodes=eval_episodes
            )
            evaluator.log_evaluation(eval_return, eval_length)
            evaluator.log_success_rate(eval_success_rate)

            continual_metrics = metrics_tracker.get_summary()
            continual_metrics["success_rate"] = eval_success_rate
            evaluator.log_continual_metrics(continual_metrics)

            pbar.set_postfix({
                "return": f"{continual_metrics['average_return']:.2f}",
                "success": f"{eval_success_rate:.2%}",
                "forgetting": f"{continual_metrics['forgetting_measure']:.2f}",
            })

        evaluator.step()

    # Save final metrics
    final_metrics = metrics_tracker.get_summary()
    metrics_path = os.path.join(
        evaluator.get_log_dir(), "csv", "final_metrics.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save checkpoint
    if save_checkpoints:
        checkpoint_path = os.path.join(
            checkpoint_dir, method_name, f"{method_name}_continualbench_final.pt"
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        agent.save(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    evaluator.close()
    env.close()
    agent.close()

    logger.info(f"Experiment completed. Results saved to {evaluator.get_log_dir()}")
    logger.info(f"Final metrics: {final_metrics}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="ContinualBench Benchmark Runner")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=[
            "all", "base", "fast_trac", "parseval",
            "dual_learner", "world_model", "dual_actor"
        ],
        help="Method to run (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run (overrides --seed)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000000,
        help="Total training steps (default: 1M)",
    )
    parser.add_argument(
        "--steps-per-task",
        type=int,
        default=100000,
        help="Steps per task",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10000,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Episodes per evaluation",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Disable saving checkpoints",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("ContinualBench Benchmark")
    logger.info("=" * 60)

    # Check if ContinualBench is available
    if not CONTINUALBENCH_AVAILABLE:
        logger.error("ContinualBench not installed!")
        logger.error("Install with:")
        logger.error("  git clone https://github.com/sail-sg/ContinualBench.git")
        logger.error("  cd ContinualBench && pip install -e .")
        sys.exit(1)

    # Determine methods to run
    if args.method == "all":
        methods = [
            "base", "fast_trac", "parseval",
            "dual_learner", "world_model", "dual_actor"
        ]
    else:
        methods = [args.method]

    # Determine seeds to run
    seeds = args.seeds if args.seeds is not None else [args.seed]

    # Run experiments
    all_results = {}

    for method in methods:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running method: {method}")
        logger.info(f"{'='*40}")

        method_results = {}

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")

            try:
                result = run_continualbench_experiment(
                    method_name=method,
                    seed=seed,
                    total_steps=args.steps,
                    eval_interval=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    steps_per_task=args.steps_per_task,
                    checkpoint_dir=args.checkpoint_dir,
                    save_checkpoints=not args.no_save_checkpoints,
                    render=args.render,
                )
                method_results[str(seed)] = result

            except Exception as e:
                logger.error(f"Experiment failed for seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                method_results[str(seed)] = {"error": str(e)}

        all_results[method] = method_results

    # Save aggregate results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        args.output_dir, f"continualbench_results_{timestamp}.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("All experiments completed!")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
