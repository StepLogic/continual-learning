import argparse
import json
import time
import jax.numpy as jnp
from qr_dqn.configs import QRDQNConfig
from qr_dqn.train import train


def main():
    parser = argparse.ArgumentParser(description="QR-DQN on Atari")
    parser.add_argument("--game", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--num_quantiles", type=int, default=64)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=10_000_000)
    parser.add_argument("--replay_capacity", type=int, default=100_000)
    parser.add_argument("--warmup_steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_update_freq", type=int, default=1_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval_interval", type=int, default=50_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    args = parser.parse_args()

    config = QRDQNConfig(
        game=args.game,
        num_quantiles=args.num_quantiles,
        kappa=args.kappa,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_frames=args.max_frames,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )

    print(f"Training QR-DQN on {config.game}")
    print(f"Config: {config}")
    start = time.time()
    agent, metrics = train(config)
    elapsed = time.time() - start

    num_episodes = len(metrics["episode_returns"])
    mean_return = (
        float(jnp.mean(jnp.array(metrics["episode_returns"][-10:])))
        if num_episodes >= 10
        else 0.0
    )
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Episodes: {num_episodes}")
    print(f"Mean return (last 10): {mean_return:.2f}")

    results = {
        "config": vars(config),
        "num_episodes": num_episodes,
        "mean_return_last_10": mean_return,
        "training_time_s": elapsed,
    }
    with open(f"qr_dqn_results_{config.game}_{config.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to qr_dqn_results_{config.game}_{config.seed}.json")


if __name__ == "__main__":
    main()