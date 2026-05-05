import logging

import jax
import jax.numpy as jnp
import numpy as np
from qr_dqn.configs import QRDQNConfig
from qr_dqn.agent import QRDQNAgent
from qr_dqn.atari_wrapper import make_atari_env
from qr_dqn.replay import NStepBuffer
from task_mastery import TaskMasteryTracker, TaskMasteryConfig

logger = logging.getLogger("continual_rl")


def get_epsilon(frame: int, config: QRDQNConfig) -> float:
    if frame >= config.epsilon_decay_steps:
        return config.epsilon_end
    slope = (config.epsilon_end - config.epsilon_start) / config.epsilon_decay_steps
    return config.epsilon_start + slope * frame


def _log_metrics(frame, avg_loss, epsilon, episode_count, episode_returns, max_return):
    recent = episode_returns[-10:] if episode_returns else [0.0]
    mean_ret = sum(recent) / len(recent)
    max_ret_str = f"{max_return:.1f}" if max_return != float("-inf") else "--"
    print(
        f"[Frame {frame:>7d}] loss={avg_loss:.3f} | eps={epsilon:.3f} "
        f"| episodes={episode_count} | mean_ret(10)={mean_ret:.1f} | max_ret={max_ret_str}"
    )


def train(config: QRDQNConfig, max_frames_override: int = None):
    max_frames = max_frames_override or config.max_frames
    env = make_atari_env(config.game, seed=config.seed)
    rng = jax.random.PRNGKey(config.seed)
    rng, agent_rng = jax.random.split(rng)

    num_actions = env.action_space.n
    obs, _ = env.reset()
    obs_shape = obs.shape

    agent = QRDQNAgent(config, num_actions=num_actions, obs_shape=obs_shape, rng=agent_rng)

    nstep_buffer = NStepBuffer(n=config.n_step, gamma=config.gamma) if config.n_step > 1 else None

    metrics = {"losses": [], "episode_returns": [], "eval_returns": []}
    episode_return = 0.0
    episode_count = 0
    max_return = float("-inf")
    losses_since_log = []

    for frame in range(max_frames):
        epsilon = get_epsilon(frame, config)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store terminated (not truncated) for correct bootstrapping
        if nstep_buffer is not None:
            nstep_result = nstep_buffer.push(obs, action, reward, next_obs, terminated)
            if nstep_result is not None:
                obs_n, action_n, reward_n, next_obs_n, term_n = nstep_result
                agent.buffer_add(obs_n, action_n, reward_n, next_obs_n, term_n)
        else:
            agent.buffer_add(obs, action, reward, next_obs, terminated)

        episode_return += reward
        obs = next_obs

        if terminated or truncated:
            if nstep_buffer is not None:
                for obs_n, action_n, reward_n, next_obs_n, term_n in nstep_buffer.flush():
                    agent.buffer_add(obs_n, action_n, reward_n, next_obs_n, term_n)

            metrics["episode_returns"].append(episode_return)
            max_return = max(max_return, episode_return)
            logger.info(
                "Episode %d | Return: %.1f | Max Return: %.1f | Frame: %d",
                episode_count, episode_return, max_return, frame,
            )
            episode_return = 0.0
            episode_count += 1
            obs, _ = env.reset()

        if frame >= config.warmup_steps:
            rng, train_rng = jax.random.split(rng)
            train_metrics = agent.train_step(train_rng)
            metrics["losses"].append(train_metrics["loss"])
            losses_since_log.append(train_metrics["loss"])

        if config.log_interval and frame > 0 and frame % config.log_interval == 0:
            avg_loss = sum(losses_since_log) / len(losses_since_log) if losses_since_log else 0.0
            _log_metrics(frame, avg_loss, epsilon, episode_count, metrics["episode_returns"], max_return)
            losses_since_log = []

        if config.eval_interval and frame > 0 and frame % config.eval_interval == 0:
            eval_return = evaluate(agent, env, config.eval_episodes, config.seed + 1000)
            metrics["eval_returns"].append(eval_return)
            print(f"Eval @ {frame}: mean_return={eval_return:.1f} over {config.eval_episodes} episodes")

    return agent, metrics


def evaluate(agent: QRDQNAgent, env, num_episodes: int = 10, seed: int = 0) -> float:
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_return = 0.0
        done = False
        while not done:
            action = agent.act(obs, epsilon=0.001)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            done = terminated or truncated
        returns.append(total_return)
    return float(np.mean(returns))


def train_on_task(
    agent: QRDQNAgent,
    env,
    config: QRDQNConfig,
    steps_per_task: int,
    evaluator=None,
    metrics_tracker=None,
    task_idx: int = 0,
    mastery_tracker: TaskMasteryTracker = None,
):
    """
    Train agent on a single task for a fixed number of steps.

    Args:
        agent: Pre-initialized QRDQNAgent (params persist across tasks).
        env: Environment to train on.
        config: QRDQNConfig.
        steps_per_task: Number of frames to train for this task.
        evaluator: Optional Evaluator for logging.
        metrics_tracker: Optional ContinualMetrics for tracking.
        task_idx: Index of current task.

    Returns:
        Updated agent, dict of metrics.
    """
    obs, _ = env.reset()
    episode_return = 0.0
    episode_count = 0
    max_return = float("-inf")
    losses_since_log = []
    metrics = {"losses": [], "episode_returns": [], "eval_returns": []}

    nstep_buffer = NStepBuffer(n=config.n_step, gamma=config.gamma) if config.n_step > 1 else None
    rng = jax.random.PRNGKey(config.seed)

    for frame in range(steps_per_task):
        epsilon = get_epsilon(agent.step_count, config)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store terminated (not truncated) for correct bootstrapping
        if nstep_buffer is not None:
            nstep_result = nstep_buffer.push(obs, action, reward, next_obs, terminated)
            if nstep_result is not None:
                obs_n, action_n, reward_n, next_obs_n, term_n = nstep_result
                agent.buffer_add(obs_n, action_n, reward_n, next_obs_n, term_n)
        else:
            agent.buffer_add(obs, action, reward, next_obs, terminated)

        episode_return += reward
        obs = next_obs

        if terminated or truncated:
            if nstep_buffer is not None:
                for obs_n, action_n, reward_n, next_obs_n, term_n in nstep_buffer.flush():
                    agent.buffer_add(obs_n, action_n, reward_n, next_obs_n, term_n)

            metrics["episode_returns"].append(episode_return)
            max_return = max(max_return, episode_return)

            if evaluator is not None:
                task_name = info.get("task_name", getattr(env, "get_current_game", lambda: "unknown")())
                task_switched = info.get("task_switched", False)
                evaluator.log_episode(
                    episode_return=episode_return,
                    episode_length=frame,
                    task_idx=task_idx,
                    task_name=task_name,
                    task_switched=task_switched,
                )

            if metrics_tracker is not None:
                metrics_tracker.log_episode(episode_return, task_idx, evaluator.global_step if evaluator else frame)

            if mastery_tracker is not None:
                mastery_tracker.log_episode(episode_return, frame)

            episode_return = 0.0
            episode_count += 1
            obs, _ = env.reset()

        if agent.step_count >= config.warmup_steps:
            rng, train_rng = jax.random.split(rng)
            train_metrics = agent.train_step(train_rng)
            metrics["losses"].append(train_metrics["loss"])
            losses_since_log.append(train_metrics["loss"])

        if config.log_interval and frame > 0 and frame % config.log_interval == 0:
            avg_loss = sum(losses_since_log) / len(losses_since_log) if losses_since_log else 0.0
            _log_metrics(frame, avg_loss, epsilon, episode_count, metrics["episode_returns"], max_return)
            losses_since_log = []

        if config.eval_interval and frame > 0 and frame % config.eval_interval == 0:
            eval_return = evaluate(agent, env, config.eval_episodes, config.seed + 1000)
            metrics["eval_returns"].append(eval_return)
            if evaluator is not None:
                evaluator.log_evaluation(eval_return, 0)
            if mastery_tracker is not None:
                mastery_tracker.log_eval(eval_return, frame)
            print(f"Eval @ task {task_idx} frame {frame}: mean_return={eval_return:.1f}")

        # Check task mastery
        if mastery_tracker is not None and mastery_tracker.check_mastery(frame):
            print(f"\n  Task {task_idx} MASTERY ACHIEVED at step {frame} (episode {episode_count})")
            print(f"  Mean return (last 20): {mastery_tracker.get_metrics().final_mean_return:.1f}")
            print(f"  CV: {mastery_tracker.get_metrics().cv_recent:.3f}")
            break

        if evaluator is not None:
            evaluator.step()

    return agent, metrics


def evaluate_on_game(agent: QRDQNAgent, game: str, num_episodes: int = 10, seed: int = 0) -> float:
    """
    Evaluate agent on a specific Atari game.

    Creates a fresh environment for the game, runs evaluation,
    and closes the environment.
    """
    env = make_atari_env(game, seed=seed)
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_return = 0.0
        done = False
        while not done:
            action = agent.act(obs, epsilon=0.001)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            done = terminated or truncated
        returns.append(total_return)
    env.close()
    return float(np.mean(returns))