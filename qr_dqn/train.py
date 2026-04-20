import logging

import jax
import jax.numpy as jnp
import numpy as np
from qr_dqn.configs import QRDQNConfig
from qr_dqn.agent import QRDQNAgent
from qr_dqn.atari_wrapper import make_atari_env

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

    metrics = {"losses": [], "episode_returns": [], "eval_returns": []}
    episode_return = 0.0
    episode_count = 0
    max_return = float("-inf")
    losses_since_log = []

    for frame in range(max_frames):
        epsilon = get_epsilon(frame, config)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.buffer_add(obs, action, reward, next_obs, done)
        episode_return += reward
        obs = next_obs

        if done:
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