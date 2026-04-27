# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
"""
Cross-game continual learning training and evaluation for all FAME paper baselines.

Trains a single agent sequentially on:
  1. Breakout (mode 0)
  2. Freeway (modes 0-7)
  3. SpaceInvaders (modes 0-9)

Then evaluates across all 19 game/mode combos (cross-game evaluation).

All environments use ActionPadWrapper to normalize action spaces to MAX_ACTIONS=6.
Invalid actions are masked with -inf in the policy logits.

Supported baselines (FAME paper Table 1):
  cnn-simple    — Reset / from-scratch baseline
  cnn-simple-ft — Fine-tuning baseline
  cnn-componet  — CompoNet (compositional network)
  prog-net      — ProgressiveNet (progressive neural networks)
  packnet       — PackNet (pruning-based)
  fame          — FAME (fast + meta dual-learner)

Usage:
  # Single baseline (train + eval)
  python run_baselines_cross_game.py --model-type packnet --seed 1 --total-timesteps 1000000
  python run_baselines_cross_game.py --model-type fame --seed 1 --total-timesteps 1000000
  # All baselines (train + eval)
  python run_baselines_cross_game.py --model-type all --seed 1 --total-timesteps 1000000
  # Eval only (skip training, load from existing checkpoints)
  python run_baselines_cross_game.py --model-type packnet --eval-only --seed 1
  python run_baselines_cross_game.py --model-type all --eval-only --seed 1
"""
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from typing import Literal, Optional
from tqdm import tqdm
from scipy import stats
import ale_py

gym.register_envs(ale_py)

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from models import (
    CnnSimpleAgent,
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
    FAMEAgent,
)
from replay import expReplay_Meta

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


ALL_BASELINE_TYPES = [
    "cnn-simple",
    "cnn-simple-ft",
    "cnn-componet",
    "prog-net",
    "packnet",
    "fame",
]


@dataclass
class Args:
    # Model type (all FAME paper baselines, or "all" to run every baseline)
    model_type: Literal[
        "cnn-simple",
        "cnn-simple-ft",
        "cnn-componet",
        "prog-net",
        "packnet",
        "fame",
        "all",
    ] = "fame"
    """The baseline model type. Use 'all' to run every FAME paper baseline sequentially."""

    eval_only: bool = False
    """Skip training and run cross-game evaluation on existing checkpoints."""

    save_dir: str = "agents"
    """Directory where checkpoints are saved."""

    # Experiment arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ppo-atari-cross-game"
    wandb_entity: str = None
    capture_video: bool = False

    # Algorithm specific arguments
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

    # FAME-specific arguments
    lr_fast: float = 2.5e-4
    lr_meta: float = 2.5e-4
    size_fast2meta: int = 20000
    size_meta: int = 200000
    detection_step: int = 1200
    epoch_meta: int = 200
    warmstep: int = 50000
    lambda_reg: float = 1.0
    use_ttest: int = 1

    # CompoNet-specific arguments
    componet_finetune_encoder: bool = False
    prevs_to_noise: int = 0

    # Evaluation arguments
    num_eval_episodes: int = 10
    """Number of episodes per game/mode during cross-game evaluation."""
    max_eval_timesteps: int = 1000
    """Max timesteps per episode during cross-game evaluation."""

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


def make_env(env_id, idx, capture_video, run_name, mode=None, render_mode=None):
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
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = ActionPadWrapper(env, max_actions=MAX_ACTIONS)
        return env

    return thunk


def build_action_mask(game, device):
    """Build logits mask: 0 for valid actions, -inf for invalid."""
    valid_n = GAME_ACTION_SIZES[game]
    mask = torch.zeros(MAX_ACTIONS, device=device)
    mask[:valid_n] = 1.0
    return torch.where(mask == 1.0, torch.tensor(0.0, device=device), torch.tensor(float("-inf"), device=device))


def get_masked_action_and_value(agent, x, action_mask_logits, model_type, action=None, prevs_to_noise=0):
    """Get action with logits masked to prevent invalid actions in cross-game setting."""
    if model_type == "cnn-componet":
        p, _phi, hidden = agent.actor(x, ret_encoder_out=True, prevs_to_noise=prevs_to_noise)
        logits = torch.log(p + 1e-10) + action_mask_logits
    elif model_type == "prog-net":
        hidden = agent.encoder(x)
        logits = agent.actor(hidden) + action_mask_logits
    elif model_type == "fame":
        hidden = agent.network(x)
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
    if model_type == "cnn-componet":
        return agent.critic(agent.encoder(x))
    elif model_type == "prog-net":
        return agent.critic(agent.encoder(x))
    elif model_type == "fame":
        return agent.critic(agent.network(x))
    else:
        return agent.critic(agent.network(x))


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

    elif model_type == "fame":
        return FAMEAgent(envs, fast=True).to(device)


def train_meta(exp_replay_meta, exp_replay_fast2meta, Meta_agent, Meta_opt, Meta_scheduler, Meta_criterion2, args, mode, device):
    """Update the meta learner on meta buffer (old data) and fast2meta buffer (new data)."""
    BATCHSIZE = 64
    u_steps = (exp_replay_fast2meta.size() // BATCHSIZE) - 1

    for epoch in range(args.epoch_meta):
        for i, p_update in enumerate(range(u_steps)):
            states_meta, actions_meta = exp_replay_meta.sample()
            states_meta = states_meta.to(device)
            actions_meta = actions_meta.to(device)
            logits = Meta_agent.forward(states_meta)
            log_probs = F.log_softmax(logits, dim=-1)
            loss1 = Meta_criterion2(log_probs, actions_meta.view(-1).long())

            if i % (mode + 1) == 0 and mode > 0:
                states_fast, actions_fast = exp_replay_fast2meta.sample()
                logits = Meta_agent.forward(states_fast)
                log_probs = F.log_softmax(logits, dim=-1)
                loss2 = Meta_criterion2(log_probs, actions_fast.view(-1).long())
                loss = loss1 + loss2
            else:
                loss = loss1

            Meta_opt.zero_grad()
            loss.backward()
            Meta_opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}/{args.epoch_meta}, Meta Loss: {loss.item():.2e}, lr: {Meta_opt.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 2 == 0:
            Meta_scheduler.step()


def compute_task_id(test_game, test_mode):
    """Compute the task_id for a given game/mode pair in the sequential training order."""
    task_id = 0
    for g in GAME_SEQUENCE:
        if g == test_game:
            task_id += test_mode
            break
        task_id += len(GAME_MODES[g])
    return task_id


def eval_on_task(args, model_type, run_name, task_id, game, mode, writer, global_step_offset):
    """Evaluate a checkpoint on the task it was just trained on.

    Runs test_agent.py as a subprocess for a single game/mode pair and logs
    the result. Returns the average episodic return, or None on failure.
    """
    checkpoint_dir = f"{args.save_dir}/{run_name}/task_{task_id}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(script_dir, "test_agent.py")

    cmd = [
        sys.executable, test_agent_path,
        "--load", checkpoint_dir,
        "--algorithm", model_type if model_type != "fame" else "FAME",
        "--test-env", game,
        "--test-mode", str(mode),
        "--cross-game",
        "--num-episodes", str(args.num_eval_episodes),
        "--max-timesteps", str(args.max_eval_timesteps),
    ]

    if model_type in ("prog-net", "cnn-componet"):
        prev_units = [f"{args.save_dir}/{run_name}/task_{p}" for p in range(task_id)]
        if prev_units:
            cmd += ["--prev-units"] + prev_units

    if model_type == "packnet":
        cmd = [
            sys.executable, test_agent_path,
            "--load", f"{args.save_dir}/{run_name}/task_{task_id}",
            "--algorithm", model_type,
            "--mode", str(task_id + 1),
            "--test-env", game,
            "--test-mode", str(mode),
            "--task-checkpoint", checkpoint_dir,
            "--cross-game",
            "--num-episodes", str(args.num_eval_episodes),
            "--max-timesteps", str(args.max_eval_timesteps),
            "--csv",str(args.save_dir + "/" + run_name + f"/task_{task_id}" + "/eval_results.csv")
        ]

    game_name = game.replace("ALE/", "").replace("-v5", "")
    print(f"  Evaluating on {game_name} mode {mode} (task {task_id})...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        avg_ret = None
        for line in result.stdout.strip().split("\n"):
            if line.startswith("Avg. episodic return:"):
                avg_ret = float(line.split(":")[-1].strip())
                break
        if avg_ret is not None:
            print(f"  Per-task eval: {game_name} mode {mode} = {avg_ret:.2f}")
            writer.add_scalar(
                f"per_task_eval/{game_name}_m{mode}", avg_ret, global_step_offset
            )
            return avg_ret
        else:
            print(f"  Per-task eval FAILED (no output)")
            return None
    except subprocess.TimeoutExpired:
        print(f"  Per-task eval TIMEOUT")
        return None
    except Exception as e:
        print(f"  Per-task eval ERROR: {e}")
        return None


def run_training(args, model_type):
    """Train a single baseline on all games/modes sequentially."""
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

    is_fame = model_type == "fame"

    run_name = f"cross_game__{model_type}__{args.exp_name}__{args.seed}"
    print(f"*** Run's name: {run_name} ***")
    print(f"*** Model type: {model_type} ***")
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

    # =========================================================================
    # FAME-specific setup (only created when model_type == "fame")
    # =========================================================================
    if is_fame:
        BATCHSIZE = 64
        exp_replay_fast2meta = expReplay_Meta(max_size=args.size_fast2meta, batch_size=BATCHSIZE, device=device)
        exp_replay_meta = expReplay_Meta(max_size=args.size_meta, batch_size=BATCHSIZE, device=device)

        env_default = gym.vector.SyncVectorEnv([
            make_env("ALE/Freeway-v5", i, args.capture_video, "init", mode=0, render_mode="rgb_array")
            for i in range(args.num_envs)
        ])
        Meta_agent = FAMEAgent(env_default, fast=False).to(device)
        Meta_opt = optim.Adam(Meta_agent.parameters(), lr=args.lr_meta, eps=1e-5)
        Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
        Meta_criterion2 = torch.nn.NLLLoss()
        warmstep = int(args.warmstep / args.batch_size)
        fast2meta_step = int(args.size_fast2meta / args.batch_size)
        MAX_STEP = 100

    # =========================================================================
    # Training state
    # =========================================================================
    agent = None
    optimizer = None
    prev_checkpoint_paths = []
    packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)
    global_step_offset = 0  # cumulative step count across tasks for TensorBoard x-axis
    per_task_eval_results = []  # (task_id, game, mode, avg_return)

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

            envs = gym.vector.SyncVectorEnv([
                make_env(GAME, i, args.capture_video, run_name, mode=mode, render_mode="rgb_array")
                for i in range(args.num_envs)
            ])
            assert isinstance(envs.single_action_space, gym.spaces.Discrete)
            assert envs.single_action_space.n == MAX_ACTIONS

            action_mask_logits = build_action_mask(GAME, device)

            is_first_task = (task_id == 0)

            # --- FAME: detection step at task boundary ---
            META_WARMUP = 0
            if is_fame and not is_first_task:
                print("  Running FAME detection...")
                epi_return_fast = 0
                epi_return_meta = 0
                avereward_fast = []
                avereward_meta = []
                max_step = 0

                currnt_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)

                for step_small in range(args.detection_step):
                    action, _, _, _ = get_masked_action_and_value(
                        agent, currnt_obs / 255.0, action_mask_logits, model_type
                    )
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

                max_step = 0
                currnt_obs, _ = envs.reset(seed=args.seed)
                currnt_obs = torch.Tensor(currnt_obs).to(device)

                for step_small in range(args.detection_step):
                    action, _, _, _ = get_masked_action_and_value(
                        Meta_agent, currnt_obs / 255.0, action_mask_logits, model_type
                    )
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

            # --- Agent creation / update at task boundary ---
            needs_new_agent = model_type in ["cnn-componet", "prog-net"]
            needs_reload = model_type in ["cnn-simple-ft", "packnet"]

            if is_first_task:
                if is_fame:
                    agent = FAMEAgent(envs, fast=True).to(device)
                    optimizer = optim.Adam(agent.parameters(), lr=args.lr_fast, eps=1e-5)
                else:
                    agent = create_agent(
                        model_type, envs, [], device, args,
                        packnet_task_id=1 if model_type == "packnet" else None,
                        is_first_task=True,
                    )
                    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            elif needs_new_agent:
                ckpt_path = f"{args.save_dir}/{run_name}/task_{task_id - 1}"
                agent.save(ckpt_path)
                prev_checkpoint_paths.append(ckpt_path)

                agent = create_agent(
                    model_type, envs, prev_checkpoint_paths, device, args,
                    is_first_task=False,
                )
                optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            elif needs_reload:
                ckpt_path = f"{args.save_dir}/{run_name}/task_{task_id - 1}"
                agent.save(ckpt_path)
                prev_checkpoint_paths.append(ckpt_path)

                if model_type == "cnn-simple-ft":
                    agent = create_agent(
                        model_type, envs, [ckpt_path], device, args,
                        is_first_task=False,
                    )
                    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

                elif model_type == "packnet":
                    packnet_task_id = task_id + 1
                    agent.start_retraining()
                    agent.network.task_id = packnet_task_id
                    agent.actor = layer_init(nn.Linear(512, MAX_ACTIONS), std=0.01)
                    agent.critic = layer_init(nn.Linear(512, 1), std=1)
                    agent.retrain_mode = False
                    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            elif is_fame:
                pass

            # Rollout buffers
            obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs)).to(device)

            global_step = global_step_offset
            start_time = time.time()
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)

            for iteration in tqdm(range(1, args.num_iterations + 1), desc=f"{game_name}_m{mode}"):
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / args.num_iterations
                    lrnow = frac * (args.lr_fast if is_fame else args.learning_rate)
                    optimizer.param_groups[0]["lr"] = lrnow

                for step in range(args.num_steps):
                    global_step += args.num_envs
                    obs[step] = next_obs
                    dones[step] = next_done

                    with torch.no_grad():
                        action, logprob, _, value = get_masked_action_and_value(
                            agent, next_obs / 255.0, action_mask_logits, model_type,
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

                    if is_fame and global_step > fast2meta_step:
                        rollout_obs = obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:])
                        rollout_action = actions.reshape(actions.shape[0] * actions.shape[1], *actions.shape[2:])
                        exp_replay_fast2meta.store(rollout_obs, rollout_action)

                    if "final_info" in infos:
                        for info in infos["final_info"]:
                            if info and "episode" in info:
                                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                with torch.no_grad():
                    next_value = get_masked_value(agent, next_obs / 255.0, model_type).reshape(1, -1)
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

                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                b_inds = np.arange(args.batch_size)
                clipfracs = []
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, args.batch_size, args.minibatch_size):
                        end = start + args.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = get_masked_action_and_value(
                            agent, b_obs[mb_inds] / 255.0, action_mask_logits, model_type,
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

                        if is_fame and META_WARMUP == 1 and args.lambda_reg > 0 and iteration < warmstep:
                            with torch.no_grad():
                                logits_meta = Meta_agent.forward(b_obs[mb_inds] / 255.0)
                                logits_meta = logits_meta + action_mask_logits
                                soft_target = F.softmax(logits_meta, dim=-1)
                            logits_agent = agent.forward(b_obs[mb_inds] / 255.0)
                            logits_agent = logits_agent + action_mask_logits
                            logit_input = F.log_softmax(logits_agent, dim=-1)
                            loss_reg = F.kl_div(logit_input, soft_target, reduction='batchmean')

                            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.lambda_reg * loss_reg
                        else:
                            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        if model_type == "packnet":
                            if global_step >= packnet_retrain_start:
                                agent.start_retraining()
                            agent.before_update()
                        optimizer.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # --- End of task: post-task updates ---
            ckpt_path = f"{args.save_dir}/{run_name}/task_{task_id}"
            agent.save(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            # --- Per-task evaluation: verify the agent learned the current task ---
            avg_ret = eval_on_task(
                args, model_type, run_name, task_id, GAME, mode, writer, global_step
            )
            per_task_eval_results.append((task_id, GAME, mode, avg_ret))

            if is_fame:
                if game_idx > 0 or mode > 0:
                    print(f"  Updating meta learner (task {task_id})...")
                    train_meta(exp_replay_meta, exp_replay_fast2meta, Meta_agent, Meta_opt, Meta_scheduler, Meta_criterion2, args, mode, device)
                    exp_replay_fast2meta.copy_to(exp_replay_meta)
                    exp_replay_fast2meta.delete()
                    print(f"  Meta buffer size: {exp_replay_meta.size()}")
                else:
                    print("  First task: skipping meta update, copying fast2meta to meta")
                    exp_replay_fast2meta.copy_to(exp_replay_meta)
                    exp_replay_fast2meta.delete()

            envs.close()
            global_step_offset = global_step
            task_id += 1

    # --- Final save for FAME ---
    if is_fame:
        agent.save(dirname=f"{args.save_dir}/{run_name}")
        Meta_agent.save(dirname=f"{args.save_dir}/{run_name}")
        env_default.close()

    writer.close()

    # --- Per-task evaluation summary ---
    per_task_csv = f"eval_per_task_{model_type}_seed{args.seed}.csv"
    with open(per_task_csv, "w") as f:
        f.write("algorithm,task_id,environment,mode,seed,avg_return\n")
        for tid, game, mode, ret in per_task_eval_results:
            f.write(f"{model_type},{tid},{game},{mode},{args.seed},{ret if ret is not None else 'NaN'}\n")

    print("\n" + "=" * 60)
    print("PER-TASK EVALUATION SUMMARY (learned verification)")
    print("-" * 60)
    for tid, game, mode, ret in per_task_eval_results:
        game_name = game.replace("ALE/", "").replace("-v5", "")
        ret_str = f"{ret:.2f}" if ret is not None else "FAILED"
        print(f"  Task {tid:2d} | {game_name:15} mode {mode:2d} | avg_return = {ret_str}")
    print(f"  Saved to: {per_task_csv}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("CROSS-GAME TRAINING COMPLETE")
    print(f"Model type: {model_type}")
    print(f"Checkpoints: {args.save_dir}/{run_name}/task_0/ ... task_{TOTAL_TASKS - 1}/")
    print("=" * 60)

    return run_name


def run_cross_game_eval(args, model_type, run_name):
    """Evaluate a trained baseline across all game/mode combos and write CSV."""
    model_base = f"{args.save_dir}/{run_name}"
    if not os.path.isdir(model_base):
        print(f"ERROR: Checkpoint directory not found: {model_base}")
        return

    eval_csv = f"eval_cross_game_{model_type}_seed{args.seed}.csv"
    print(f"\nCROSS-GAME EVALUATION: {model_type}")
    print(f"Model: {model_base}")
    print(f"Output: {eval_csv}")
    print("")

    header = "algorithm,environment,train_mode,test_mode,seed,ep_ret"
    with open(eval_csv, "w") as f:
        f.write(header + "\n")

    total_eval = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(script_dir, "test_agent.py")

    for test_game in GAME_SEQUENCE:
        test_game_name = test_game.replace("ALE/", "").replace("-v5", "")
        modes = GAME_MODES[test_game]

        print(f"Evaluating on {test_game} ({len(modes)} modes)...")

        for test_mode in modes:
            task_id = compute_task_id(test_game, test_mode)
            checkpoint_dir = f"{model_base}/task_{task_id}"

            cmd = [
                sys.executable, test_agent_path,
                "--load", checkpoint_dir,
                "--algorithm", model_type if model_type != "fame" else "FAME",
                "--test-env", test_game,
                "--test-mode", str(test_mode),
                "--cross-game",
                "--num-episodes", str(args.num_eval_episodes),
                "--max-timesteps", str(args.max_eval_timesteps),
            ]

            # Prog-net and CompoNet need prev-units
            if model_type in ("prog-net", "cnn-componet"):
                prev_units = []
                for prev in range(task_id):
                    prev_units.append(f"{model_base}/task_{prev}")
                if prev_units:
                    cmd += ["--prev-units"] + prev_units

            # PackNet: load from final checkpoint with correct task view
            if model_type == "packnet":
                cmd = [
                    sys.executable, test_agent_path,
                    "--load", f"{model_base}/task_{TOTAL_TASKS - 1}",
                    "--algorithm", model_type,
                    "--mode", str(task_id),
                    "--test-env", test_game,
                    "--test-mode", str(test_mode),
                    "--task-checkpoint", checkpoint_dir,
                    "--cross-game",
                    "--num-episodes", str(args.num_eval_episodes),
                    "--max-timesteps", str(args.max_eval_timesteps),
                ]

            # FAME: use fast agent
            if model_type == "fame":
                cmd = [
                    sys.executable, test_agent_path,
                    "--load", checkpoint_dir,
                    "--algorithm", "FAME",
                    "--test-env", test_game,
                    "--test-mode", str(test_mode),
                    "--cross-game",
                    "--num-episodes", str(args.num_eval_episodes),
                    "--max-timesteps", str(args.max_eval_timesteps),
                ]

            print(f"  Mode {test_mode} (task {task_id}): ", end="", flush=True)

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                avg_ret = None
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Avg. episodic return:"):
                        avg_ret = line.split(":")[-1].strip()
                        break
                if avg_ret is not None:
                    print(avg_ret)
                    with open(eval_csv, "a") as f:
                        f.write(f"{model_type},{test_game},{task_id},{test_mode},{args.seed},{avg_ret}\n")
                    total_eval += 1
                else:
                    print("FAILED (no output)")
                    with open(eval_csv, "a") as f:
                        f.write(f"{model_type},{test_game},{task_id},{test_mode},{args.seed},NaN\n")
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                with open(eval_csv, "a") as f:
                    f.write(f"{model_type},{test_game},{task_id},{test_mode},{args.seed},NaN\n")
            except Exception as e:
                print(f"ERROR: {e}")
                with open(eval_csv, "a") as f:
                    f.write(f"{model_type},{test_game},{task_id},{test_mode},{args.seed},NaN\n")

    print(f"\nEvaluation complete: {total_eval} successful evaluations")
    print(f"Results saved to: {eval_csv}")
    return eval_csv


def print_eval_summary(eval_csv):
    """Print a summary table from an evaluation CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(eval_csv)
        print(f"\nResults Summary ({eval_csv}):")
        print("-" * 50)
        for game in df["environment"].unique():
            game_data = df[df["environment"] == game]["ep_ret"]
            game_name = str(game).replace("ALE/", "").replace("-v5", "")
            print(f"  {game_name:15}: {game_data.mean():7.2f} (+/- {game_data.std():.2f})")
    except ImportError:
        pass


if __name__ == "__main__":
    args = tyro.cli(Args)

    baselines = ALL_BASELINE_TYPES if args.model_type == "all" else [args.model_type]

    all_eval_csvs = []
    for i, model_type in enumerate(baselines):
        if args.model_type == "all":
            print(f"\n{'#' * 60}")
            print(f"# Baseline {i + 1}/{len(baselines)}: {model_type}")
            print(f"{'#' * 60}")

        # Training
        if not args.eval_only:
            run_name = run_training(args, model_type)
        else:
            run_name = f"cross_game__{model_type}__{args.exp_name}__{args.seed}"

        # Cross-game evaluation
        eval_csv = run_cross_game_eval(args, model_type, run_name)
        if eval_csv:
            all_eval_csvs.append(eval_csv)
            print_eval_summary(eval_csv)

    # Combined summary when running all baselines
    if len(all_eval_csvs) > 1:
        print(f"\n{'=' * 60}")
        print("COMBINED RESULTS SUMMARY")
        print(f"{'=' * 60}")
        try:
            import pandas as pd
            dfs = []
            for csv_file in all_eval_csvs:
                try:
                    dfs.append(pd.read_csv(csv_file))
                except Exception:
                    pass
            if dfs:
                all_df = pd.concat(dfs, ignore_index=True)
                print("\nAverage Return by Algorithm and Game:")
                print("-" * 60)
                for algo in all_df["algorithm"].unique():
                    algo_df = all_df[all_df["algorithm"] == algo]
                    print(f"\n  {algo}:")
                    for game in algo_df["environment"].unique():
                        game_data = algo_df[algo_df["environment"] == game]["ep_ret"]
                        game_name = str(game).replace("ALE/", "").replace("-v5", "")
                        print(f"    {game_name:15}: {game_data.mean():7.2f} (+/- {game_data.std():.2f})")
        except ImportError:
            pass