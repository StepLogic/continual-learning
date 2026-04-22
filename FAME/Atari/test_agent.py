import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import ale_py

gym.register_envs(ale_py)

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from models import (
    CnnSimpleAgent,
    DinoSimpleAgent,
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
)
from task_utils import parse_name_info, path_from_other_mode

# Cross-game: max action space across all games
MAX_ACTIONS = 6


class ActionPadWrapper(gym.ActionWrapper):
    """Pads the action space to max_actions for cross-game compatibility."""

    def __init__(self, env, max_actions):
        super().__init__(env)
        self.valid_actions = env.action_space.n
        self.action_space = gym.spaces.Discrete(max_actions)

    def action(self, action):
        return int(np.clip(action, 0, self.valid_actions - 1))


def parse_arguments():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--load", type=str, required=True)

    parser.add_argument("--algorithm", type=str, default=None,
                        choices=["cnn-simple", "cnn-simple-ft", "dino-simple",
                                 "cnn-componet", "prog-net", "packnet", "FAME"],
                        help="Override algorithm (default: parsed from directory name)")
    parser.add_argument("--prev-units", type=str, nargs='+', default=[],
                        help="Paths to previous model checkpoints (for prog-net, cnn-componet)")
    parser.add_argument("--task-checkpoint", type=str, default=None,
                        help="Path to per-task checkpoint for loading actor/critic (packnet cross-game)")
    parser.add_argument("--cross-game", default=False, action='store_true',
                        help="Pad action space to MAX_ACTIONS for cross-game evaluation")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", type=int, default=None)

    # Cross-game evaluation: override test environment (e.g., train on Freeway, test on Breakout)
    parser.add_argument("--test-env", type=str, default=None,
                        choices=["ALE/Breakout-v5", "ALE/SpaceInvaders-v5", "ALE/Freeway-v5"],
                        help="Test on different game (default: use train game)")
    parser.add_argument("--test-mode", type=int, default=None,
                        help="Test on different mode (default: use train mode)")

    parser.add_argument("--max-timesteps", type=int, default=1000)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--csv', default=None, type=str)
    # fmt: on

    return parser.parse_args()


def make_env(env_id, idx, run_name, render_mode=None, mode=None, dino=False, cross_game=False):
    def thunk():
        env = gym.make(env_id, mode=mode, render_mode=render_mode)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        # env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)

        if not dino:
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
        else:
            env = gym.wrappers.ResizeObservation(env, (224, 224))
        env = gym.wrappers.FrameStackObservation(env, 4)
        if cross_game:
            env = ActionPadWrapper(env, max_actions=MAX_ACTIONS)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_arguments()

    if args.algorithm is not None:
        algorithm = args.algorithm
        _, seed = parse_name_info(args.load.split("/")[-1])
    else:
        algorithm, seed = parse_name_info(args.load.split("/")[-1])

    # Use override if provided, otherwise default to trained values
    test_env = args.test_env
    test_mode = args.test_mode
    train_mode=test_mode
    mode = args.mode
    env_name= test_env
    seed = seed if args.seed is None else args.seed

    print(
        f"\nTrain: {env_name} | Test: {test_env} mode {test_mode}, algorithm: {algorithm}, seed: {seed}\n"
    )

    # make the environment (with ActionPadWrapper for cross-game eval)
    dino = "dino" in algorithm
    envs = gym.vector.SyncVectorEnv([make_env(test_env, 1, run_name="test", mode=test_mode, dino=dino, cross_game=args.cross_game)])
    env_fn = make_env(
        test_env,
        0,
        run_name="test",
        mode=test_mode,
        render_mode="human" if args.render else None,
        dino=dino,
        cross_game=args.cross_game,
    )
    env = env_fn()

    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if algorithm in ["cnn-simple", "cnn-simple-ft"]:
        agent = CnnSimpleAgent.load(
            args.load, envs, load_critic=False, map_location=device
        )
    elif algorithm == "cnn-componet":
        if args.prev_units:
            prevs_paths = args.prev_units
        else:
            prevs_paths = [path_from_other_mode(args.load, i) for i in range(mode)]
        agent = CnnCompoNetAgent.load(
            args.load, envs, prevs_paths=prevs_paths, map_location=device
        )
    elif algorithm == "packnet":
        task_id = None if args.mode == None else args.mode + 1
        agent = PackNetAgent.load(args.load, task_id=task_id, map_location=device)
        agent.network.set_view(task_id)

        if args.task_checkpoint:
            # Cross-game: load actor/critic from per-task checkpoint
            ac = PackNetAgent.load(args.task_checkpoint, map_location=device)
            agent.critic = ac.critic
            agent.actor = ac.actor
        elif mode != train_mode:
            # load the actor and critic heads from the model trained in the testing task (game mode)
            path = path_from_other_mode(args.load, mode)
            ac = PackNetAgent.load(path, map_location=device)
            agent.critic = ac.critic
            agent.actor = ac.actor
    elif algorithm == "dino-simple":
        from models import DinoSimpleAgent
        agent = DinoSimpleAgent.load(args.load, envs, dino_size="s", frame_stack=4, device=device)
    elif algorithm == "prog-net":
        if args.prev_units:
            prevs_paths = args.prev_units
        else:
            prevs_paths = [path_from_other_mode(args.load, i) for i in range(mode)]
        agent = ProgressiveNetAgent.load(
            args.load, envs, prevs_paths=prevs_paths, map_location=device
        )
    elif algorithm == "FAME":
        from models import FAMEAgent
        agent = FAMEAgent.load(args.load, envs, fast=True, map_location=device)
    else:
        print(f"Loading of agent type `{algorithm}` is not implemented.")
        quit(1)

    agent.to(device)

    #
    # Main loop
    # ~~~~~~~~~
    # Build action mask for cross-game eval: mask invalid actions with -inf
    valid_actions = envs.single_action_space.n
    if algorithm == "cnn-componet":
        model_actions = agent.actor.out_dim
    else:
        model_actions = agent.actor.out_features
    if model_actions > valid_actions:
        action_mask = torch.zeros(model_actions, device=device)
        action_mask[:valid_actions] = 1.0
        action_mask_logits = torch.where(action_mask == 1.0, torch.tensor(0.0, device=device), torch.tensor(float('-inf'), device=device))
    else:
        action_mask_logits = torch.zeros(model_actions, device=device)

    ep_rets = []
    for _ in range(args.num_episodes):
        observation, info = env.reset(seed=seed)

        ep_ret = 0
        for _ in range(args.max_timesteps):
            observation = torch.from_numpy(np.array(observation)).to(device) / 255.0
            observation = observation.unsqueeze(0)

            # Get action with masking for cross-game compatibility
            if algorithm == "dino-simple":
                hidden = agent.middle(agent.dino.encode(observation * 255.0))
                logits = agent.actor(hidden) + action_mask_logits
            elif algorithm == "cnn-componet":
                p, _phi, hidden = agent.actor(observation, ret_encoder_out=True)
                logits = torch.log(p + 1e-10) + action_mask_logits
            elif algorithm == "prog-net":
                hidden = agent.encoder(observation)
                logits = agent.actor(hidden) + action_mask_logits
            else:  # cnn-simple, cnn-simple-ft, packnet, FAME
                hidden = agent.network(observation)
                logits = agent.actor(hidden) + action_mask_logits
            action = logits.argmax(dim=-1)  # deterministic

            observation, reward, terminated, truncated, info = env.step(
                action[0].item()
            )

            ep_ret += reward

            if terminated or truncated:
                print(f"Episodic return: {ep_ret}")
                ep_rets.append(ep_ret)
                break

    print()
    print("Avg. episodic return:", np.mean(ep_rets))

    if args.csv:
        exists = os.path.exists(args.csv)
        with open(args.csv, "w" if not exists else "a") as f:
            if not exists:
                f.write("algorithm,environment,train mode,test mode,seed,ep ret\n")
            for v in ep_rets:
                f.write(f"{algorithm},{env_name},{train_mode},{mode},{seed},{v}\n")
