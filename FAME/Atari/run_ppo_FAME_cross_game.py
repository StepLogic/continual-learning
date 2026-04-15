# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
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
from replay import *
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy
from scipy import stats
import ale_py

# from replay import expReplay_Meta

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs


from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from models import FAMEAgent


# Games to train on sequentially
GAME_SEQUENCE = ["ALE/Breakout-v5", "ALE/Freeway-v5", "ALE/SpaceInvaders-v5"]
GAME_MODES = {
    "ALE/Breakout-v5": [0],
    "ALE/Freeway-v5": list(range(8)),
    "ALE/SpaceInvaders-v5": list(range(10)),
}
# Action space sizes per game (for cross-game compatibility)
GAME_ACTION_SIZES = {
    "ALE/Breakout-v5": 4,
    "ALE/Freeway-v5": 3,
    "ALE/SpaceInvaders-v5": 6,
}
# Use the maximum action space across all games
MAX_ACTIONS = max(GAME_ACTION_SIZES.values())  # 6 (SpaceInvaders)


@dataclass
class Args:
    # Model type
    model_type: Literal["FAME"] = "FAME"
    """The name of the model to use as agent."""

    save_dir: str = 'agents'
    """Directory where the trained model will be saved."""

    # Experiment arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ppo-atari-cross-game"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    total_timesteps: int = int(1e6)
    """total timesteps per task"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # FAME-specific arguments
    lr_fast: float = 2.5e-4
    lr_meta: float = 2.5e-4
    size_fast2meta: int = 20000
    size_meta: int = 200000
    detection_step: int = 1200
    epoch_meta: int = 200
    warmstep: int = 50000
    lambda_reg: float = 1.0
    use_ttest: int = 1  # 1: use t-test; 0: use empirical ranking

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, mode=None, dino=False,render_mode=None):
    def thunk():
        env = gym.make(env_id, mode=mode,render_mode=render_mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        # Pad action space to MAX_ACTIONS for cross-game compatibility
        env = ActionPadWrapper(env, max_actions=MAX_ACTIONS)
        return env

    return thunk


class ActionPadWrapper(gym.ActionWrapper):
    """Pads the action space to max_actions by masking invalid actions.
    Only passes through valid actions (0..n-1) to the underlying env.
    Invalid actions (n..max-1) have zero probability in the policy via masking."""

    def __init__(self, env, max_actions):
        super().__init__(env)
        self.valid_actions = env.action_space.n
        self.action_space = gym.spaces.Discrete(max_actions)
        # Create a mask: 1 for valid, -inf for invalid
        self.action_mask = np.zeros(max_actions, dtype=np.float32)
        self.action_mask[:self.valid_actions] = 1.0

    def action(self, action):
        # Clip action to valid range (safety)
        return int(np.clip(action, 0, self.valid_actions - 1))


def train_meta(exp_replay_meta, exp_replay_fast2meta, Meta_agent, Meta_opt, Meta_scheduler, Meta_criterion2, args, mode, device):
    """Update the meta learner on meta buffer (old data) and fast2meta buffer (new data)."""
    BATCHSIZE = 64
    u_steps = (exp_replay_fast2meta.size() // BATCHSIZE) - 1

    for epoch in range(args.epoch_meta):
        for i, p_update in enumerate(range(u_steps)):
            # Step 1: update meta learner via old data from meta buffer
            states_meta, actions_meta = exp_replay_meta.sample()
            states_meta = states_meta.to(device)
            actions_meta = actions_meta.to(device)
            logits = Meta_agent.forward(states_meta)
            log_probs = F.log_softmax(logits, dim=-1)
            loss1 = Meta_criterion2(log_probs, actions_meta.view(-1).long())

            # Step 2: update meta learner via new data from fast2meta buffer (only every gameid time)
            if i % (mode + 1) == 0 and mode > 0:
                states_fast, actions_fast = exp_replay_fast2meta.sample()
                logits = Meta_agent.forward(states_fast)
                log_probs = F.log_softmax(logits, dim=-1)
                loss2 = Meta_criterion2(log_probs, actions_fast.view(-1).long())
                loss = loss1 + loss2
            else:
                loss = loss1

            # Step 3: update the meta learner
            Meta_opt.zero_grad()
            loss.backward()
            Meta_opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}/{args.epoch_meta}, Meta Loss: {loss.item():.2e}, lr: {Meta_opt.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 2 == 0:
            Meta_scheduler.step()


def train_agent(agent, optimizer, obs, actions, logprobs, rewards, dones, values, returns, b_inds, args, device, writer, global_step, start_time):
    """Train the fast agent (PPO + optional behavior cloning regularization)."""
    # Optimizing the policy and value network
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                obs[mb_inds] / 255.0, actions.long()[mb_inds]
            )
            logratio = newlogprob - logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = returns[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(newvalue - values[mb_inds], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    # Record losses
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


def get_masked_action_and_value(agent, x, action_mask_logits, action=None):
    """Get action with logits masked to prevent invalid actions in cross-game setting."""
    hidden = agent.network(x)
    logits = agent.actor(hidden)
    # Mask invalid actions with -inf so they get zero probability
    logits = logits + action_mask_logits
    probs = torch.distributions.Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    logprob = probs.log_prob(action)
    entropy = probs.entropy()
    value = agent.critic(hidden)
    return action, logprob, entropy, value


def get_masked_value(agent, x):
    """Get value (action masking doesn't affect the critic)."""
    return agent.critic(agent.network(x))


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Compute batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # FAME-specific setup
    BATCHSIZE = 64
    exp_replay_fast2meta = expReplay_Meta(max_size=args.size_fast2meta, batch_size=BATCHSIZE, device=device)
    exp_replay_meta = expReplay_Meta(max_size=args.size_meta, batch_size=BATCHSIZE, device=device)

    # Create env with padded action space (MAX_ACTIONS = 6)
    env_default = gym.vector.SyncVectorEnv([
        make_env("ALE/Freeway-v5", i, args.capture_video, "init", mode=0,render_mode="rgb_array")
        for i in range(args.num_envs)
    ])

    # Initialize FAME agents with MAX_ACTIONS output (PERSIST ACROSS ALL GAMES)
    agent = FAMEAgent(env_default, fast=True).to(device)
    Meta_agent = FAMEAgent(env_default, fast=False).to(device)
    random_agent = FAMEAgent(env_default, fast=True).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr_fast, eps=1e-5)
    Meta_opt = optim.Adam(Meta_agent.parameters(), lr=args.lr_meta, eps=1e-5)
    Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
    Meta_criterion2 = torch.nn.NLLLoss()

    warmstep = int(args.warmstep / args.batch_size)
    fast2meta_step = int(args.size_fast2meta / args.batch_size)
    MAX_STEP = 100

    # Run name for final model
    run_name = f"cross_game__FAME__{args.exp_name}__{args.seed}"
    print(f"*** Run's name: {run_name} ***")

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

    # =========================================================================
    # TRAINING LOOP: Sequential across GAMES
    # =========================================================================
    game_idx = 0
    for GAME in GAME_SEQUENCE:
        game_name = GAME.replace("ALE/", "").replace("-v5", "")
        mode_list = GAME_MODES[GAME]
        num_modes = len(mode_list)

        print("")
        print("=" * 60)
        print(f"TRAINING ON {GAME} ({num_modes} modes) - Game {game_idx + 1}/{len(GAME_SEQUENCE)}")
        print("=" * 60)

        for mode in mode_list:
            print(f"\n--- Mode {mode}/{num_modes - 1} ---")

            # Create environment for this mode (action-padded to MAX_ACTIONS)
            envs = gym.vector.SyncVectorEnv([
                make_env(GAME, i, args.capture_video, run_name, mode=mode,render_mode="rgb_array")
                for i in range(args.num_envs)
            ])
            assert isinstance(envs.single_action_space, gym.spaces.Discrete)
            assert envs.single_action_space.n == MAX_ACTIONS

            # Build action mask for this game (mask invalid actions with -inf)
            valid_n = GAME_ACTION_SIZES[GAME]
            action_mask = torch.zeros(MAX_ACTIONS, device=device)
            action_mask[:valid_n] = 1.0
            action_mask_logits = torch.where(action_mask == 1.0, torch.tensor(0.0, device=device), torch.tensor(float('-inf'), device=device))

            # Rollout buffers
            obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs)).to(device)

            # Start training
            global_step = 0
            start_time = time.time()
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)

            # Detection step for FAME (skip for first game/mode)
            META_WARMUP = 0
            if game_idx > 0 or mode > 0:
                # Policy evaluation to choose initialization
                print("  Running FAME detection...")
                epi_return_fast = 0
                epi_return_meta = 0
                avereward_fast = []
                avereward_meta = []
                max_step = 0

                currnt_obs = next_obs.clone()

                # Evaluate fast learner
                for step_small in range(args.detection_step):
                    action, _, _, _ = get_masked_action_and_value(agent, currnt_obs / 255.0, action_mask_logits)
                    next_obs_eval, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
                    epi_return_fast += reward
                    currnt_obs = torch.Tensor(next_obs_eval).to(device)
                    done = np.logical_or(terminations, truncations)
                    max_step += 1
                    if done.all() or max_step > MAX_STEP:
                        currnt_obs, _ = envs.reset(seed=args.seed)
                        currnt_obs = torch.Tensor(currnt_obs).to(device)
                        avereward_fast.append(epi_return_fast)
                        epi_return_fast = 0
                        max_step = 0

                # Evaluate meta learner
                max_step = 0
                currnt_obs, _ = envs.reset(seed=args.seed)
                currnt_obs = torch.Tensor(currnt_obs).to(device)

                for step_small in range(args.detection_step):
                    action, _, _, _ = get_masked_action_and_value(Meta_agent, currnt_obs / 255.0, action_mask_logits)
                    next_obs_eval, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
                    epi_return_meta += reward
                    currnt_obs = torch.Tensor(next_obs_eval).to(device)
                    done = np.logical_or(terminations, truncations)
                    max_step += 1
                    if done.all() or max_step > MAX_STEP:
                        currnt_obs, _ = envs.reset(seed=args.seed)
                        currnt_obs = torch.Tensor(currnt_obs).to(device)
                        avereward_meta.append(epi_return_meta)
                        epi_return_meta = 0
                        max_step = 0

                # Decide initialization
                Avereward_fast = np.mean(avereward_fast) if avereward_fast else -1000
                Avereward_meta = np.mean(avereward_meta) if avereward_meta else -1000

                print(f"  Fast avg: {Avereward_fast:.2f}, Meta avg: {Avereward_meta:.2f}")

                if args.use_ttest == 1 and len(avereward_fast) >= 2 and len(avereward_meta) >= 2:
                    _, p_value = stats.ttest_ind(avereward_meta, avereward_fast, alternative='greater', equal_var=False)
                    META_WARMUP = 1 if p_value.any() < 0.05 else 0
                else:
                    META_WARMUP = 1 if Avereward_meta > Avereward_fast else 0

                if META_WARMUP:
                    print("  Using META initialization with behavior cloning regularization")
                else:
                    print("  Using FAST initialization (fine-tuning)")

            # Main PPO training loop
            for iteration in tqdm(range(1, args.num_iterations + 1), desc=f"Mode {mode}"):
                # Annealing learning rate
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / args.num_iterations
                    lrnow = frac * args.lr_fast
                    optimizer.param_groups[0]["lr"] = lrnow

                # Rollout
                for step in range(args.num_steps):
                    global_step += args.num_envs
                    obs[step] = next_obs
                    dones[step] = next_done

                    with torch.no_grad():
                        action, logprob, _, value = get_masked_action_and_value(agent, next_obs / 255.0, action_mask_logits)
                        values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob

                    next_obs, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
                    next_done = np.logical_or(terminations, truncations)
                    rewards[step] = torch.tensor(reward).to(device).view(-1)
                    next_obs = torch.Tensor(next_obs).to(device)
                    next_done = torch.Tensor(next_done).to(device)

                    # Store in fast2meta buffer for meta-learning
                    if global_step > fast2meta_step:
                        rollout_obs = obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:])
                        rollout_action = actions.reshape(actions.shape[0] * actions.shape[1], *actions.shape[2:])
                        exp_replay_fast2meta.store(rollout_obs, rollout_action)

                    # Log episodic returns
                    if "final_info" in _:
                        for info in _["final_info"]:
                            if info and "episode" in info:
                                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                # Bootstrap value
                with torch.no_grad():
                    next_value = get_masked_value(agent, next_obs / 255.0).reshape(1, -1)
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
                b_inds = np.arange(args.batch_size)

                # PPO update (with optional BC regularization during warmup)
                if META_WARMUP == 1 and args.lambda_reg > 0 and iteration < warmstep:
                    # Add behavior cloning loss
                    for epoch in range(args.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, args.batch_size, args.minibatch_size):
                            end = start + args.minibatch_size
                            mb_inds = b_inds[start:end]

                            _, newlogprob, entropy, newvalue = get_masked_action_and_value(
                                agent, b_obs[mb_inds] / 255.0, action_mask_logits, b_actions.long()[mb_inds]
                            )
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            # BC regularization
                            with torch.no_grad():
                                logits_meta = Meta_agent.forward(b_obs[mb_inds] / 255.0)
                                logits_meta = logits_meta + action_mask_logits  # mask meta logits too
                                soft_target = F.softmax(logits_meta, dim=-1)
                            logits_agent = agent.forward(b_obs[mb_inds] / 255.0)
                            logits_agent = logits_agent + action_mask_logits  # mask agent logits
                            logit_input = F.log_softmax(logits_agent, dim=-1)
                            loss_reg = F.kl_div(logit_input, soft_target, reduction='batchmean')

                            # PPO losses
                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            newvalue = newvalue.view(-1)
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                            entropy_loss = entropy.mean()

                            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.lambda_reg * loss_reg

                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                            optimizer.step()
                else:
                    # Standard PPO update
                    for epoch in range(args.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, args.batch_size, args.minibatch_size):
                            end = start + args.minibatch_size
                            mb_inds = b_inds[start:end]

                            _, newlogprob, entropy, newvalue = get_masked_action_and_value(
                                agent, b_obs[mb_inds] / 255.0, action_mask_logits, b_actions.long()[mb_inds]
                            )
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            newvalue = newvalue.view(-1)
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                            entropy_loss = entropy.mean()

                            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                            optimizer.step()

                # Record metrics
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # End of mode: update meta learner
            if game_idx > 0 or mode > 0:
                print(f"  Updating meta learner (mode {mode})...")
                train_meta(exp_replay_meta, exp_replay_fast2meta, Meta_agent, Meta_opt, Meta_scheduler, Meta_criterion2, args, mode, device)
                exp_replay_fast2meta.copy_to(exp_replay_meta)
                exp_replay_fast2meta.delete()
                print(f"  Meta buffer size: {exp_replay_meta.size()}")
            else:
                print("  First game/mode: skipping meta update, just copying fast2meta to meta")
                exp_replay_fast2meta.copy_to(exp_replay_meta)
                exp_replay_fast2meta.delete()

            envs.close()

        game_idx += 1

    # Save final model
    if args.save_dir is not None:
        print(f"\nSaving final model to {args.save_dir}/{run_name}")
        agent.save(dirname=f"{args.save_dir}/{run_name}")
        Meta_agent.save(dirname=f"{args.save_dir}/{run_name}")

    writer.close()
    env_default.close()

    print("\n" + "=" * 60)
    print("CROSS-GAME TRAINING COMPLETE")
    print(f"Final model saved: {args.save_dir}/{run_name}")
    print("=" * 60)
