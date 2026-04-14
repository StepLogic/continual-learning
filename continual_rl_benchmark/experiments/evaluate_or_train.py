"""
Evaluate or Train Script for Continual RL Benchmark.

For each task:
1. Check if checkpoint exists at checkpoints/agent_name/agent_name_task{idx}.pt
2. If checkpoint exists: load and evaluate
3. If checkpoint doesn't exist: train model for that task

Usage:
    python evaluate_or_train.py --method base --env continual_world --seeds 0 1 2
    python evaluate_or_train.py --method all --checkpoint-dir checkpoints
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from agents import (
    SACAgent,
    FastTRACAgent,
    ParsevalAgent,
    DualLearnerAgent,
    WorldModelAgent,
    DualActorAgent,
)
from environments import ContinualEnv
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
            "buffer_size": 10000,
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
            "buffer_size": 10000,
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
            "buffer_size": 10000,
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
            "buffer_size": 10000,
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
            "buffer_size": 10000,
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
            "buffer_size": 10000,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        },
    }
    return configs.get(method_name, configs["base"])


def get_checkpoint_path(checkpoint_dir: str, agent_name: str, task_idx: int) -> str:
    """Get checkpoint path for a given task."""
    return os.path.join(checkpoint_dir, agent_name, f"{agent_name}_task{task_idx}.pt")


def checkpoint_exists(checkpoint_dir: str, agent_name: str, task_idx: int) -> bool:
    """Check if checkpoint exists for a given task."""
    path = get_checkpoint_path(checkpoint_dir, agent_name, task_idx)
    return os.path.exists(path)


def evaluate(
    agent,
    env,
    num_episodes: int = 10,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate agent on current task.

    Returns average return and length across episodes.
    """
    returns = []
    lengths = []

    for _ in range(num_episodes):
        obs = env.reset()
        episode_return = 0
        episode_length = 0

        while True:
            action = agent.get_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            if done:
                break

        returns.append(episode_return)
        lengths.append(episode_length)

    return np.mean(returns), np.mean(lengths)


def train_for_task(
    agent,
    env: ContinualEnv,
    task_idx: int,
    steps_per_task: int,
    eval_interval: int,
    eval_episodes: int,
    evaluator: Evaluator,
    metrics_tracker: ContinualMetrics,
    logger,
) -> dict:
    """
    Train agent for a single task.

    Returns final metrics for the task.
    """
    obs = env.reset()
    episode_return = 0
    episode_length = 0

    warmup_steps = min(10000, steps_per_task // 100)
    pbar = tqdm(range(steps_per_task), desc=f"Training task {task_idx}")

    for step in pbar:
        # Get action
        if step < warmup_steps:
            action = agent.get_action(obs, random=True)
        else:
            action = agent.get_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
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
            task_idx_current = info.get("task_idx", 0)
            task_name = info.get("task_name", "unknown")
            task_switched = info.get("task_switched", False)

            evaluator.log_episode(
                episode_return=episode_return,
                episode_length=episode_length,
                task_idx=task_idx_current,
                task_name=task_name,
                task_switched=task_switched,
                additional_metrics=update_metrics if update_metrics else None,
            )

            metrics_tracker.log_episode(episode_return, task_idx_current, evaluator.global_step)

            if task_switched:
                metrics_tracker.end_task(task_idx_current)

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
            eval_return, eval_length = evaluate(agent, env, num_episodes=eval_episodes)
            evaluator.log_evaluation(eval_return, eval_length)

            continual_metrics = metrics_tracker.get_summary()
            evaluator.log_continual_metrics(continual_metrics)

            pbar.set_postfix({
                "return": f"{continual_metrics['average_return']:.2f}",
                "forgetting": f"{continual_metrics['forgetting_measure']:.2f}",
            })

        evaluator.step()

    return metrics_tracker.get_summary()


def process_task(
    method_name: str,
    task_idx: int,
    checkpoint_dir: str,
    env: ContinualEnv,
    steps_per_task: int,
    eval_interval: int,
    eval_episodes: int,
    seed: int,
    logger,
    results_dir: str = "results",
) -> dict:
    """
    Process a single task: evaluate if checkpoint exists, otherwise train.

    Returns task metrics.
    """
    # Check for checkpoint
    has_checkpoint = checkpoint_exists(checkpoint_dir, method_name, task_idx)

    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create agent
    config = get_agent_config(method_name)
    AgentClass = AGENT_REGISTRY.get(method_name, SACAgent)
    agent = AgentClass(obs_dim=obs_dim, act_dim=act_dim, **config)

    # Create evaluator for this task
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{method_name}_task{task_idx}_{timestamp}"
    evaluator = Evaluator(experiment_name=experiment_name, seed=seed)

    # Create metrics tracker
    metrics_tracker = ContinualMetrics(num_tasks=env.get_num_tasks())

    if has_checkpoint:
        # Load checkpoint and evaluate
        checkpoint_path = get_checkpoint_path(checkpoint_dir, method_name, task_idx)
        logger.info(f"Loading checkpoint for task {task_idx}: {checkpoint_path}")

        try:
            agent.load(checkpoint_path)
            logger.info(f"Successfully loaded checkpoint for task {task_idx}")

            # Evaluate
            logger.info(f"Evaluating agent on task {task_idx}...")
            eval_return, eval_length = evaluate(
                agent, env, num_episodes=eval_episodes, deterministic=True
            )

            logger.info(
                f"Task {task_idx} evaluation complete: "
                f"return={eval_return:.2f}, length={eval_length:.2f}"
            )

            result = {
                "status": "evaluated",
                "checkpoint_path": checkpoint_path,
                "eval_return": eval_return,
                "eval_length": eval_length,
            }

        except Exception as e:
            logger.error(f"Failed to load checkpoint for task {task_idx}: {e}")
            result = {
                "status": "load_failed",
                "error": str(e),
            }

    else:
        # Train for this task
        logger.info(
            f"No checkpoint found for task {task_idx}. "
            f"Training for {steps_per_task} steps..."
        )

        # Set environment to this specific task
        while env.current_task_idx != task_idx:
            env._switch_task()

        task_metrics = train_for_task(
            agent=agent,
            env=env,
            task_idx=task_idx,
            steps_per_task=steps_per_task,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            evaluator=evaluator,
            metrics_tracker=metrics_tracker,
            logger=logger,
        )

        # Save checkpoint after training
        checkpoint_path = evaluator.save_task_checkpoint(
            agent, method_name, task_idx, checkpoint_dir
        )
        logger.info(f"Saved checkpoint for task {task_idx}: {checkpoint_path}")

        final_metrics = metrics_tracker.get_summary()
        result = {
            "status": "trained",
            "checkpoint_path": checkpoint_path,
            "metrics": final_metrics,
        }

    evaluator.close()
    agent.close()

    return result


def run_evaluation_or_training(
    method_name: str,
    env_name: str,
    seed: int,
    checkpoint_dir: str,
    steps_per_task: int,
    eval_interval: int,
    eval_episodes: int,
    smoke_test: bool = False,
) -> dict:
    """
    Run evaluation or training for all tasks.

    Returns results for all tasks.
    """
    logger = get_logger()
    logger.info(
        f"Starting evaluate_or_train: {method_name} on {env_name} (seed={seed})"
    )

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create environment
    steps_per_task = 5000 if smoke_test else steps_per_task
    env = ContinualEnv(env_name=env_name, steps_per_task=steps_per_task, seed=seed)

    num_tasks = env.get_num_tasks()
    logger.info(f"Number of tasks: {num_tasks}")

    # Create checkpoint directory
    os.makedirs(os.path.join(checkpoint_dir, method_name), exist_ok=True)

    # Process each task
    all_results = {}

    for task_idx in range(num_tasks):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing task {task_idx}/{num_tasks - 1}: {env.task_list[task_idx]}")
        logger.info(f"{'='*60}")

        try:
            result = process_task(
                method_name=method_name,
                task_idx=task_idx,
                checkpoint_dir=checkpoint_dir,
                env=env,
                steps_per_task=steps_per_task,
                eval_interval=eval_interval,
                eval_episodes=eval_episodes,
                seed=seed,
                logger=logger,
            )
            all_results[str(task_idx)] = result

            # Switch to next task for next iteration
            if task_idx < num_tasks - 1:
                env._switch_task()

        except Exception as e:
            logger.error(f"Failed to process task {task_idx}: {e}")
            all_results[str(task_idx)] = {
                "status": "failed",
                "error": str(e),
            }

    env.close()

    logger.info(f"\n{'='*60}")
    logger.info("All tasks completed!")
    logger.info(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or Train Continual RL Agents"
    )
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
        "--env",
        type=str,
        default="continual_world",
        choices=["continual_world", "procgen"],
        help="Environment to run",
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
        "--steps-per-task",
        type=int,
        default=400000,
        help="Steps per task for training",
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
        help="Directory to load/save checkpoints",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal test",
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
    logger.info("Continual RL: Evaluate or Train")
    logger.info("=" * 60)

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

    # Run for each method and seed
    all_results = {}

    for method in methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"Method: {method}")
        logger.info(f"{'='*50}")

        method_results = {}

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")

            try:
                result = run_evaluation_or_training(
                    method_name=method,
                    env_name=args.env,
                    seed=seed,
                    checkpoint_dir=args.checkpoint_dir,
                    steps_per_task=args.steps_per_task,
                    eval_interval=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    smoke_test=args.smoke_test,
                )
                method_results[str(seed)] = result

            except Exception as e:
                logger.error(f"Failed for seed {seed}: {e}")
                method_results[str(seed)] = {"error": str(e)}

        all_results[method] = method_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        args.output_dir, f"eval_or_train_{timestamp}.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
