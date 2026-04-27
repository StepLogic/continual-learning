"""
Continual Learning Evaluation Script for QR-DQN.

Trains QR-DQN sequentially on Atari games and evaluates on all seen tasks.
Usage:
    python experiments/run_qr_dqn_continual.py \
        --games Breakout SpaceInvaders Freeway \
        --steps-per-task 500000 \
        --seed 42
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax

from qr_dqn.configs import QRDQNConfig
from qr_dqn.agent import QRDQNAgent
from qr_dqn.continual_env import ContinualAtariEnv
from qr_dqn.train import train_on_task, evaluate_on_game
from qr_dqn.atari_wrapper import make_atari_env
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import ContinualMetrics
from src.utils.logging import setup_logging, get_logger


def evaluate_all_tasks(agent: QRDQNAgent, games: List[str], eval_episodes: int, seed: int) -> Dict[int, float]:
    """Evaluate agent on all tasks seen so far."""
    per_task_returns = {}
    for task_idx, game in enumerate(games):
        eval_return = evaluate_on_game(agent, game, num_episodes=eval_episodes, seed=seed)
        per_task_returns[task_idx] = eval_return
        print(f"  Task {task_idx} ({game}): return={eval_return:.1f}")
    return per_task_returns


def run_continual_experiment(
    config: QRDQNConfig,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    smoke_test: bool = False,
) -> Dict:
    """
    Run continual learning experiment with QR-DQN.

    For each task:
    1. Check if checkpoint exists -> load and evaluate
    2. Else train on task -> save checkpoint
    3. Evaluate on all seen tasks
    4. Log continual metrics

    Returns:
        Dict with task_results and continual_metrics.
    """
    logger = get_logger()
    logger.info(f"Starting continual QR-DQN experiment (seed={seed})")

    # Set seeds
    np.random.seed(seed)

    # Determine parameters
    games = list(config.games)
    steps_per_task = 500 if smoke_test else config.steps_per_task
    eval_episodes = 1 if smoke_test else config.eval_episodes
    eval_interval = 0 if smoke_test else config.eval_interval

    logger.info(f"Games: {games}")
    logger.info(f"Steps per task: {steps_per_task}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create continual environment
    env = ContinualAtariEnv(games=games, steps_per_task=steps_per_task, seed=seed)

    # Get observation and action dimensions from first environment
    first_env = make_atari_env(games[0], seed=seed)
    num_actions = first_env.action_space.n
    obs, _ = first_env.reset()
    obs_shape = obs.shape
    first_env.close()

    # Create agent
    rng = jax.random.PRNGKey(seed)
    rng, agent_rng = jax.random.split(rng)
    agent = QRDQNAgent(config, num_actions=num_actions, obs_shape=obs_shape, rng=agent_rng)

    # Create evaluator
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"qr_dqn_continual_{timestamp}"
    evaluator = Evaluator(log_dir=results_dir, experiment_name=experiment_name, seed=seed)

    # Create metrics tracker
    metrics_tracker = ContinualMetrics(num_tasks=len(games))

    # Process each task
    all_task_results = []

    for task_idx in range(len(games)):
        game = games[task_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {task_idx}/{len(games)-1}: {game}")
        logger.info(f"{'='*60}")

        # Checkpoint path
        checkpoint_path = os.path.join(checkpoint_dir, f"qr_dqn_task{task_idx}_seed{seed}.pkl")

        if os.path.exists(checkpoint_path):
            # Load checkpoint and evaluate
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            agent.load(checkpoint_path)
            logger.info(f"Checkpoint loaded. Evaluating on task {task_idx}...")

            eval_return = evaluate_on_game(agent, game, num_episodes=eval_episodes, seed=seed)
            logger.info(f"Task {task_idx} eval return: {eval_return:.2f}")

            task_result = {
                "task_idx": task_idx,
                "game": game,
                "status": "evaluated",
                "checkpoint_path": checkpoint_path,
                "eval_return": eval_return,
            }
        else:
            # Train on this task
            logger.info(f"Training on task {task_idx} for {steps_per_task} steps...")

            # Set environment to this task
            while env.current_task_idx != task_idx:
                env._switch_task()

            # Create task-specific evaluator config
            task_config = QRDQNConfig(
                **{k: v for k, v in vars(config).items() if k not in ("steps_per_task", "games")}
            )
            task_config.eval_interval = eval_interval

            agent, task_metrics = train_on_task(
                agent=agent,
                env=env,
                config=task_config,
                steps_per_task=steps_per_task,
                evaluator=evaluator,
                metrics_tracker=metrics_tracker,
                task_idx=task_idx,
            )

            # Save checkpoint
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # End task in metrics tracker
            metrics_tracker.end_task(task_idx)

            task_result = {
                "task_idx": task_idx,
                "game": game,
                "status": "trained",
                "checkpoint_path": checkpoint_path,
                "num_episodes": len(task_metrics["episode_returns"]),
                "final_eval_returns": task_metrics["eval_returns"],
            }

        # Evaluate on all seen tasks
        logger.info(f"Evaluating on all tasks 0..{task_idx}...")
        seen_games = games[: task_idx + 1]
        per_task_returns = evaluate_all_tasks(agent, seen_games, eval_episodes, seed)

        # Log continual metrics
        continual_summary = metrics_tracker.get_summary()
        evaluator.log_continual_metrics(continual_summary)

        logger.info(f"Continual metrics: {continual_summary}")

        task_result["per_task_eval_returns"] = per_task_returns
        task_result["continual_metrics"] = continual_summary
        all_task_results.append(task_result)

        # Switch to next task if not last
        if task_idx < len(games) - 1:
            if not env.is_task_switch():
                env._switch_task()

    env.close()
    evaluator.close()

    # Compile final results
    final_results = {
        "config": {k: str(v) if isinstance(v, tuple) else v for k, v in vars(config).items()},
        "seed": seed,
        "task_results": all_task_results,
        "continual_metrics": metrics_tracker.get_summary(),
    }

    # Save results
    results_path = os.path.join(results_dir, f"qr_dqn_continual_seed{seed}.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="QR-DQN Continual Learning on Atari")
    parser.add_argument("--games", type=str, nargs="+", default=["Breakout", "SpaceInvaders", "Freeway"],
                        help="Ordered list of Atari games")
    parser.add_argument("--steps-per-task", type=int, default=500_000,
                        help="Training steps per task")
    parser.add_argument("--eval-interval", type=int, default=25_000,
                        help="Steps between periodic evaluations")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes per evaluation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save/load checkpoints")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run minimal test")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints")

    args = parser.parse_args()

    setup_logging(level="INFO")
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("QR-DQN Continual Learning Evaluation")
    logger.info("=" * 60)

    config = QRDQNConfig(
        games=tuple(args.games),
        steps_per_task=args.steps_per_task,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    results = run_continual_experiment(
        config=config,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        smoke_test=args.smoke_test,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Experiment complete!")
    logger.info(f"Final continual metrics: {results['continual_metrics']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
