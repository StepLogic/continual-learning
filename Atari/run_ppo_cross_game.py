# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
"""
Cross-game continual learning training for PPO-based baselines.

Trains a single agent sequentially on:
  1. Breakout (mode 0)
  2. Freeway (modes 0-7)
  3. SpaceInvaders (modes 0-9)

All environments use ActionPadWrapper to normalize action spaces to MAX_ACTIONS=6.
Invalid actions are masked with -inf in the policy logits.

Supported baselines:
  cnn-simple, cnn-simple-ft, dino-simple, cnn-componet, prog-net, packnet

Usage:
  python run_ppo_cross_game.py --model-type packnet --seed 1 --total-timesteps 1000000
"""
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from typing import Literal, Tuple, Optional
import pathlib
from tqdm import tqdm
import ale_py

gym.register_envs(ale_py)


from torch.utils.tensorboard import SummaryWriter

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


# Games to train on sequentially
GAME_SEQUENCE = ["ALE/Breakout-v5", "ALE/Freeway-v5", "ALE/SpaceInvaders-v5"]
GAME_MODES = {
    "ALE/Breakout-v5": [0],
    "ALE/Freeway-v5": list(range(8)),
    "ALE/SpaceInvaders-v5": list(range(10)),
}
GAME_ACTION_SIZES = {
    "ALE/Breakout-v5": 4,
    "ALE/Freeway-v5": 3,
    "ALE/SpaceInvaders-v5": 6,
}
MAX_ACTIONS = max(GAME_ACTION_SIZES.values())  # 6 (SpaceInvaders)
TOTAL_TASKS = sum(len(modes) for modes in GAME_MODES.values())  # 19


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass
class Args:
    model_type: Literal[
        "cnn-simple",
        "cnn-simple-ft",
        "dino-simple",
        "cnn-componet",
        "prog-net",
        "packnet",
    ] = "packnet"
    """The baseline model type."""

    dino_size: Literal["s", "b", "l", "g"] = "s"
    """Size of the dino model (only for dino-simple)."""
    componet_finetune_encoder: bool = False
    """Whether to finetune CompoNet's encoder from the previous task."""
    prevs_to_noise: Optional[int] = 0
    """Number of previous CompoNet modules to replace with noise (ablation)."""

    save_dir: str = "agents"
    """Directory where checkpoints are saved."""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ppo-atari-cross-game"
    wandb_entity: str = None
    capture_video: bool = False

    total_timesteps: int = int(1e6)
    """Timesteps per task."""
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class ActionPadWrapper(gym.ActionWrapper):
    """Pads the action space to max_actions. Clips invalid actions at env step."""

    def __init__(self, env, max_actions):
        super().__init__(env)
        self.valid_actions = env.action_space.n
        self.action_space = gym.spaces.Discrete(max_actions)

    def action(self, action):
        return int(np.clip(action, 0, self.valid_actions - 1))


def make_env(env_id, idx, capture_video, run_name, mode=None, dino=False, render_mode=None):
    def thunk():
        env = gym.make(env_id, mode=mode, render_mode=render_mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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
        env = ActionPadWrapper(env, max_actions=MAX_ACTIONS)
        return env

    return thunk


def get_masked_action_and_value(agent, x, action_mask_logits, model_type, action=None, prevs_to_noise=0):
    """Get action with logits masked to prevent invalid actions in cross-game setting."""
    if model_type == "cnn-componet":
        p, _phi, hidden = agent.actor(x, ret_encoder_out=True, prevs_to_noise=prevs_to_noise)
        # CompoNet returns probabilities; convert to logits for masking
        logits = torch.log(p + 1e-10) + action_mask_logits
    elif model_type == "dino-simple":
        hidden = agent.middle(agent.dino.encode(x * 255.0))
        logits = agent.actor(hidden) + action_mask_logits
    elif model_type == "prog-net":
        hidden = agent.encoder(x)
        logits = agent.actor(hidden) + action_mask_logits
    else:  # cnn-simple, cnn-simple-ft, packnet
        hidden = agent.network(x)
        logits = agent.actor(hidden) + action_mask_logits

    probs = torch.distributions.Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    logprob = probs.log_prob(action)
    entropy = probs.entropy()
    value = agent.critic(hidden)
    return action, logprob, entropy, value


def get_masked_value(agent, x, model_type):
    """Get value estimate (action masking doesn't affect the critic)."""
    if model_type == "dino-simple":
        return agent.critic(agent.middle(agent.dino.encode(x * 255.0)))
    elif model_type == "cnn-componet":
        return agent.critic(agent.encoder(x))
    elif model_type == "prog-net":
        return agent.critic(agent.encoder(x))
    else:
        return agent.critic(agent.network(x))


def build_action_mask(game, device):
    """Build logits mask: 0 for valid actions, -inf for invalid."""
    valid_n = GAME_ACTION_SIZES[game]
    mask = torch.zeros(MAX_ACTIONS, device=device)
    mask[:valid_n] = 1.0
    return torch.where(mask == 1.0, torch.tensor(0.0, device=device), torch.tensor(float("-inf"), device=device))


def create_agent(model_type, envs, prevs_paths, device, args, packnet_task_id=None, is_first_task=True):
    """Create or reload an agent for the given baseline."""
    if model_type == "cnn-simple":
        return CnnSimpleAgent(envs).to(device)

    elif model_type == "cnn-simple-ft":
        if is_first_task or len(prevs_paths) == 0:
            return CnnSimpleAgent(envs).to(device)
        else:
            return CnnSimpleAgent.load(
                prevs_paths[-1], envs, load_critic=False, reset_actor=True
            ).to(device)

    elif model_type == "dino-simple":
        return DinoSimpleAgent(
            envs, dino_size=args.dino_size, frame_stack=4, device=device
        ).to(device)

    elif model_type == "cnn-componet":
        return CnnCompoNetAgent(
            envs,
            prevs_paths=prevs_paths,
            finetune_encoder=args.componet_finetune_encoder,
            map_location=device,
        ).to(device)

    elif model_type == "prog-net":
        return ProgressiveNetAgent(
            envs, prevs_paths=prevs_paths, map_location=device
        ).to(device)

    elif model_type == "packnet":
        if is_first_task:
            return PackNetAgent(
                envs,
                task_id=packnet_task_id,
                is_first_task=True,
                total_task_num=TOTAL_TASKS,
            ).to(device)
        else:
            return PackNetAgent.load(
                prevs_paths[-1],
                task_id=packnet_task_id,
                restart_actor_critic=True,
                freeze_bias=True,
            ).to(device)


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    dino = "dino" in args.model_type

    run_name = f"cross_game__{args.model_type}__{args.exp_name}__{args.seed}"
    print(f"*** Run's name: {run_name} ***")
    print(f"*** Model type: {args.model_type} ***")
    print(f"*** Total tasks: {TOTAL_TASKS} ({args.total_timesteps} timesteps each) ***")

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    os.makedirs(f"{args.save_dir}/{run_name}", exist_ok=True)

    # Training state
    agent = None
    optimizer = None
    prev_checkpoint_paths = []
    packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)

    # =========================================================================
    # TRAINING LOOP: Sequential across GAMES and MODES
    # =========================================================================
    task_id = 0

    for game_idx, GAME in enumerate(GAME_SEQUENCE):
        game_name = GAME.replace("ALE/", "").replace("-v5", "")
        mode_list = GAME_MODES[GAME]
        num_modes = len(mode_list)

        print("")
        print("=" * 60)
        print(f"TRAINING ON {GAME} ({num_modes} modes) - Game {game_idx + 1}/{len(GAME_SEQUENCE)}")
        print("=" * 60)

        for mode_idx, mode in enumerate(mode_list):
            print(f"\n--- {game_name} mode {mode} (task {task_id + 1}/{TOTAL_TASKS}) ---")

            # Create environment for this mode (action-padded to MAX_ACTIONS)
            envs = gym.vector.SyncVectorEnv([
                make_env(GAME, i, args.capture_video, run_name, mode=mode, dino=dino)
                for i in range(args.num_envs)
            ])
            assert isinstance(envs.single_action_space, gym.spaces.Discrete)
            assert envs.single_action_space.n == MAX_ACTIONS

            action_mask_logits = build_action_mask(GAME, device)

            is_first_task = (task_id == 0)

            # --- Agent creation / update at task boundary ---
            needs_new_agent = args.model_type in ["cnn-componet", "prog-net"]
            needs_reload = args.model_type in ["cnn-simple-ft", "packnet"]

            if is_first_task:
                agent = create_agent(
                    args.model_type, envs, [], device, args,
                    packnet_task_id=1 if args.model_type == "packnet" else None,
                    is_first_task=True,
                )
                optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            elif needs_new_agent:
                # Save previous agent checkpoint (for prog-net/componet prev_units)
                ckpt_path = f"{args.save_dir}/{run_name}/task_{task_id - 1}"
                agent.save(ckpt_path)
                prev_checkpoint_paths.append(ckpt_path)

                agent = create_agent(
                    args.model_type, envs, prev_checkpoint_paths, device, args,
                    is_first_task=False,
                )
                optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            elif needs_reload:
                # Save previous agent checkpoint
                ckpt_path = f"{args.save_dir}/{run_name}/task_{task_id - 1}"
                agent.save(ckpt_path)
                prev_checkpoint_paths.append(ckpt_path)

                if args.model_type == "cnn-simple-ft":
                    agent = create_agent(
                        args.model_type, envs, [ckpt_path], device, args,
                        is_first_task=False,
                    )
                    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

                elif args.model_type == "packnet":
                    # Increment packnet task_id (1-based), start retraining, reset heads
                    packnet_task_id = task_id + 1
                    agent.start_retraining()  # prune current task + set_view
                    agent.network.task_id = packnet_task_id
                    agent.actor = layer_init(nn.Linear(512, MAX_ACTIONS), std=0.01)
                    agent.critic = layer_init(nn.Linear(512, 1), std=1)
                    agent.retrain_mode = False
                    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            # else: cnn-simple, dino-simple — continue with same agent

            # Rollout buffers
            obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs)).to(device)

            global_step = 0
            start_time = time.time()
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)

            for iteration in tqdm(range(1, args.num_iterations + 1), desc=f"{game_name}_m{mode}"):
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / args.num_iterations
                    lrnow = frac * args.learning_rate
                    optimizer.param_groups[0]["lr"] = lrnow

                # Rollout
                for step in range(args.num_steps):
                    global_step += args.num_envs
                    obs[step] = next_obs
                    dones[step] = next_done

                    with torch.no_grad():
                        action, logprob, _, value = get_masked_action_and_value(
                            agent, next_obs / 255.0, action_mask_logits, args.model_type,
                            prevs_to_noise=args.prevs_to_noise,
                        )
                        values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob

                    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                    next_done = np.logical_or(terminations, truncations)
                    rewards[step] = torch.tensor(reward).to(device).view(-1)
                    next_obs = torch.Tensor(next_obs).to(device)
                    next_done = torch.Tensor(next_done).to(device)

                    if "final_info" in infos:
                        for info in infos["final_info"]:
                            if info and "episode" in info:
                                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                # Bootstrap value
                with torch.no_grad():
                    next_value = get_masked_value(agent, next_obs / 255.0, args.model_type).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values

                # Flatten batch
                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # PPO update
                b_inds = np.arange(args.batch_size)
                clipfracs = []
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, args.batch_size, args.minibatch_size):
                        end = start + args.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = get_masked_action_and_value(
                            agent, b_obs[mb_inds] / 255.0, action_mask_logits, args.model_type,
                            action=b_actions.long()[mb_inds],
                            prevs_to_noise=args.prevs_to_noise,
                        )
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        newvalue = newvalue.view(-1)
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        if args.model_type == "packnet":
                            if global_step >= packnet_retrain_start:
                                agent.start_retraining()
                            agent.before_update()
                        optimizer.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                # Record metrics
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # Save checkpoint after this task
            ckpt_path = f"{args.save_dir}/{run_name}/task_{task_id}"
            agent.save(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            envs.close()
            task_id += 1

    writer.close()

    print("\n" + "=" * 60)
    print("CROSS-GAME TRAINING COMPLETE")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoints: {args.save_dir}/{run_name}/task_0/ ... task_{TOTAL_TASKS - 1}/")
    print("=" * 60)