#!/usr/bin/env python3
"""
Evaluate a QR-DQN checkpoint on a single Atari game.

Usage:
    python experiments/eval_qr_dqn_game.py \
        --load checkpoints/qr_dqn_task2_seed1.pkl \
        --game ALE/Breakout-v5 \
        --mode 0 \
        --num-episodes 10
"""
import argparse
import os
import jax
from qr_dqn.configs import QRDQNConfig
from qr_dqn.agent import QRDQNAgent
from qr_dqn.atari_wrapper import make_atari_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate QR-DQN on a single Atari game")
    parser.add_argument("--load", type=str, required=True, help="Path to checkpoint .pkl")
    parser.add_argument("--game", type=str, required=True, help="Game ID (e.g., ALE/Breakout-v5)")
    parser.add_argument("--mode", type=int, default=None, help="Game mode (optional)")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-timesteps", type=int, default=1000, help="Max timesteps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--csv", type=str, default=None, help="Append raw episode results to CSV file")
    args = parser.parse_args()

    env = make_atari_env(args.game, seed=args.seed, mode=args.mode)
    num_actions = env.action_space.n
    obs, _ = env.reset()
    obs_shape = obs.shape
    env.close()

    rng = jax.random.PRNGKey(args.seed)
    agent = QRDQNAgent(QRDQNConfig(), num_actions=num_actions, obs_shape=obs_shape, rng=rng)
    agent.load(args.load)

    returns = []
    for ep in range(args.num_episodes):
        env = make_atari_env(args.game, seed=args.seed + ep, mode=args.mode)
        obs, _ = env.reset(seed=args.seed + ep)
        total_return = 0.0
        steps = 0
        done = False
        while not done and steps < args.max_timesteps:
            action = agent.act(obs, epsilon=0.001)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            done = terminated or truncated
            steps += 1
        returns.append(total_return)
        env.close()

    avg_return = float(sum(returns) / len(returns))
    std_return = float((sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5)
    print(f"Avg. episodic return: {avg_return:.2f}")

    if args.csv:
        exists = os.path.exists(args.csv)
        with open(args.csv, "a" if exists else "w") as f:
            if not exists:
                f.write("algorithm,environment,train mode,test mode,seed,ep ret\n")
            mode_str = str(args.mode) if args.mode is not None else "0"
            for v in returns:
                f.write(f"qr_dqn,{args.game},all,{mode_str},{args.seed},{v}\n")

    return avg_return


if __name__ == "__main__":
    main()
