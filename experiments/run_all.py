"""
Continual RL Benchmark Experiment Runner.

Runs all configured methods on all environments and saves results.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

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
            "buffer_size":  10000,
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
            "buffer_size":  10000,
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
            "buffer_size":  10000,
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
            "buffer_size":  100000,
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
            "buffer_size":  10000,
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


def run_experiment(
    method_name: str,
    env_name: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    steps_per_task: int = 400000,
    smoke_test: bool = False,
    checkpoint_dir: str = "checkpoints",
    save_checkpoints: bool = True,
):
    """
    Run a single experiment.

    Args:
        method_name: Name of the method to run
        env_name: Name of the environment
        seed: Random seed
        total_steps: Total training steps
        eval_interval: Steps between evaluations
        eval_episodes: Episodes per evaluation
        steps_per_task: Steps per task (default: 400k for proper training)
        smoke_test: If True, run minimal steps for testing
    """
    logger = get_logger()
    logger.info(f"Starting experiment: {method_name} on {env_name} (seed={seed})")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create environment
    # 400,000 steps per task for proper training
    steps_per_task = 5000 if smoke_test else steps_per_task
    env = ContinualEnv(env_name=env_name, steps_per_task=steps_per_task, seed=seed)

    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create agent
    config = get_agent_config(method_name)
    AgentClass = AGENT_REGISTRY.get(method_name, SACAgent)
    agent = AgentClass(obs_dim=obs_dim, act_dim=act_dim, **config)

    # Create evaluator
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{method_name}_{env_name}_{timestamp}"
    evaluator = Evaluator(experiment_name=experiment_name, seed=seed)

    # Create metrics tracker
    metrics_tracker = ContinualMetrics(num_tasks=env.get_num_tasks())

    # Training loop
    obs = env.reset()
    episode_return = 0
    episode_length = 0

    steps = total_steps if not smoke_test else 100000
    pbar = tqdm(range(steps), desc=f"Training {method_name}")

    # Random action warmup - fill replay buffer with diverse experiences
    warmup_steps = min(10000, steps // 100)  # 1% of training or 10k steps

    for step in pbar:
        # Get action - use random actions during warmup
        if step < warmup_steps:
            action = agent.get_action(obs, random=True)
        else:
            action = agent.get_action(obs)
        # print("Action",action)
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
            # Log episode
            task_idx = info.get("task_idx", 0)
            task_name = info.get("task_name", "unknown")
            task_switched = info.get("task_switched", False)

            evaluator.log_episode(
                episode_return=episode_return,
                episode_length=episode_length,
                task_idx=task_idx,
                task_name=task_name,
                task_switched=task_switched,
                additional_metrics=update_metrics if update_metrics else None,
            )

            # Update metrics tracker
            metrics_tracker.log_episode(episode_return, task_idx, evaluator.global_step)

            if task_switched:
                metrics_tracker.end_task(task_idx)

                # Save checkpoint at task boundary
                if save_checkpoints:
                    prev_task_idx = task_idx - 1 if task_idx > 0 else env.get_num_tasks() - 1
                    checkpoint_path = evaluator.save_task_checkpoint(
                        agent, method_name, prev_task_idx, checkpoint_dir
                    )
                    logger.info(f"Saved checkpoint for task {prev_task_idx}: {checkpoint_path}")

            # Reset environment after episode ends
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
            eval_return, eval_length = evaluate(
                agent, env, num_episodes=eval_episodes
            )
            evaluator.log_evaluation(eval_return, eval_length)

            # Log continual metrics
            continual_metrics = metrics_tracker.get_summary()
            evaluator.log_continual_metrics(continual_metrics)

            pbar.set_postfix({
                "return": f"{continual_metrics['average_return']:.2f}",
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

    # Save final task checkpoint
    if save_checkpoints:
        final_task_idx = env.current_task_idx
        final_checkpoint_path = evaluator.save_task_checkpoint(
            agent, method_name, final_task_idx, checkpoint_dir
        )
        logger.info(f"Saved final checkpoint for task {final_task_idx}: {final_checkpoint_path}")

    # Save agent checkpoint (legacy location)
    checkpoint_path = os.path.join(evaluator.get_log_dir(), "agent_final.pt")
    evaluator.save_checkpoint(agent, checkpoint_path)

    evaluator.close()
    env.close()
    agent.close()

    logger.info(f"Experiment completed. Results saved to {evaluator.get_log_dir()}")
    logger.info(f"Final metrics: {final_metrics}")

    return final_metrics


def evaluate(agent, env, num_episodes=10):
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
            action = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            if done:
                break

        returns.append(episode_return)
        lengths.append(episode_length)

    return np.mean(returns), np.mean(lengths)


def main():
    parser = argparse.ArgumentParser(description="Continual RL Benchmark Runner")
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
        "--seed", type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds to run (for multiple seeds)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4000000,
        help="Total training steps (default: 4M = 10 tasks x 400k)",
    )
    parser.add_argument(
        "--steps-per-task",
        type=int,
        default=400000,
        help="Steps per task (default: 400k for proper training)",
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
        "--smoke-test",
        action="store_true",
        help="Run minimal test (10k steps, dummy env)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run full benchmark (all methods, 5 seeds)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving task checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable saving task-level checkpoints",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("Continual RL Benchmark")
    logger.info("=" * 60)

    # Determine methods to run
    if args.method == "all" or args.full_benchmark:
        methods = [
            "base", "fast_trac", "parseval",
            "dual_learner", "world_model", "dual_actor"
        ]
    else:
        methods = [args.method]

    # Determine seeds to run
    if args.full_benchmark:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [args.seed]

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
                result = run_experiment(
                    method_name=method,
                    env_name=args.env,
                    seed=seed,
                    total_steps=args.steps,
                    eval_interval=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    steps_per_task=args.steps_per_task,
                    smoke_test=False,
                    checkpoint_dir=args.checkpoint_dir,
                    save_checkpoints=not args.no_checkpoints,
                )
                method_results[str(seed)] = result

            except Exception as e:
                logger.error(f"Experiment failed for seed {seed}: {e}")
                method_results[str(seed)] = {"error": str(e)}

        all_results[method] = method_results

    # Save aggregate results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        args.output_dir, f"aggregate_results_{timestamp}.json"
    )
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("All experiments completed!")
    logger.info(f"Aggregate results saved to: {results_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
