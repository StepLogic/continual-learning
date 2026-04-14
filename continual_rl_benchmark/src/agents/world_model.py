"""
Continual RL by Planning with Online World Models
ICML 2025

Learns an online world model and acts by planning via Model Predictive Control (MPC).
Constructs unified world dynamics to handle catastrophic forgetting.
Uses SquashedGaussianPolicy for correct SAC log_prob calculation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Tuple, List
from .sac import SACAgent, ReplayBuffer, SquashedGaussianPolicy, MLP


class WorldModel(nn.Module):
    """
    Dynamics model that predicts next state and reward.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Next state prediction
        self.next_state = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward prediction
        self.reward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Done prediction
        self.done = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Predict next state, reward, and done."""
        x = self.shared(torch.cat([obs, act], dim=-1))
        next_obs = self.next_state(x)
        rew = self.reward(x)
        done = self.done(x)
        return next_obs, rew, done

    def prediction_loss(
        self, obs: torch.Tensor, act: torch.Tensor,
        next_obs: torch.Tensor, rew: torch.Tensor, done: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute prediction losses."""
        pred_next_obs, pred_rew, pred_done = self.forward(obs, act)

        obs_loss = F.mse_loss(pred_next_obs, next_obs)
        rew_loss = F.mse_loss(pred_rew, rew)
        done_loss = F.binary_cross_entropy(pred_done.squeeze(), done.squeeze())

        return {
            "obs_loss": obs_loss,
            "rew_loss": rew_loss,
            "done_loss": done_loss,
            "total_loss": obs_loss + rew_loss + done_loss,
        }


class MPCPlanner:
    """
    Model Predictive Control planner using random shooting.
    """

    def __init__(
        self,
        world_model: WorldModel,
        horizon: int = 10,
        num_samples: int = 100,
        device: str = "cuda",
    ):
        self.world_model = world_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device

    def plan(self, obs: np.ndarray, act_dim: int, act_limits: Tuple[float, float] = (-1, 1)) -> np.ndarray:
        """
        Plan action sequence using random shooting MPC.

        Returns the first action of the best sequence.
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Sample action sequences
        action_seqs = torch.FloatTensor(
            np.random.uniform(
                act_limits[0], act_limits[1],
                (self.num_samples, self.horizon, act_dim)
            )
        ).to(self.device)

        best_return = float("-inf")
        best_action = None

        with torch.no_grad():
            for i in range(self.num_samples):
                curr_obs = obs_tensor
                total_reward = 0.0

                for t in range(self.horizon):
                    act = action_seqs[i, t:t+1]
                    next_obs, rew, done = self.world_model(curr_obs, act)
                    total_reward += rew.item()
                    curr_obs = next_obs

                    if done.item() > 0.5:
                        break

                if total_reward > best_return:
                    best_return = total_reward
                    best_action = action_seqs[i, 0].cpu().numpy()

        return best_action if best_action is not None else np.zeros(act_dim)


class WorldModelAgent(SACAgent):
    """
    SAC agent with online world model and MPC planning.

    Uses the world model for:
    1. Data augmentation (generate synthetic transitions)
    2. Planning via MPC for action selection
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        model_hidden_dim: int = 512,
        model_lr: float = 0.0003,
        mpc_horizon: int = 10,
        mpc_samples: int = 100,
        model_update_freq: int = 10,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cuda",
        **kwargs
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.model_update_freq = model_update_freq

        # Replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        # SAC networks - use SquashedGaussianPolicy for correct log_prob
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q1_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.policy = SquashedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(self.device)

        self._copy_target_networks()

        # World model
        self.world_model = WorldModel(obs_dim, act_dim, model_hidden_dim).to(self.device)
        self.model_optimizer = Adam(self.world_model.parameters(), lr=model_lr)

        # MPC planner
        self.mpc_planner = MPCPlanner(
            self.world_model,
            horizon=mpc_horizon,
            num_samples=mpc_samples,
            device=self.device,
        )

        # Optimizers
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=learning_rate
        )
        self.policy_optimizer = Adam(self.policy.parameters(), lr=learning_rate)

        self.total_steps = 0
        self.use_mpc = True  # Use MPC for action selection

    def _copy_target_networks(self):
        for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_target.data.copy_(p.data)
        for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_target.data.copy_(p.data)

    def _update_target_networks(self):
        for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_target.data.lerp_(p.data, self.tau)
        for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_target.data.lerp_(p.data, self.tau)

    def get_action(self, obs: np.ndarray, deterministic: bool = False, random: bool = False) -> np.ndarray:
        """Get action using MPC planning or policy."""
        if random:
            return np.random.uniform(-1, 1, size=self.act_dim).astype(np.float32)

        if self.use_mpc and self.total_steps > 1000:
            return self.mpc_planner.plan(obs, self.act_dim)
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, _ = self.policy(obs_tensor, deterministic=deterministic)
            return action.detach().cpu().numpy()[0]

    def store_transition(
        self, obs: np.ndarray, act: np.ndarray,
        rew: float, next_obs: np.ndarray, done: bool
    ):
        """Store transition and update world model."""
        self.replay_buffer.store(obs, act, rew, next_obs, done)

        if self.total_steps % self.model_update_freq == 0:
            self._update_world_model()

    def _update_world_model(self):
        """Train world model on real transitions."""
        if self.replay_buffer.size < self.batch_size:
            return

        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        obs2 = torch.FloatTensor(batch["obs2"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch["done"]).unsqueeze(1).to(self.device)

        losses = self.world_model.prediction_loss(obs, act, obs2, rew, done)

        self.model_optimizer.zero_grad()
        losses["total_loss"].backward()
        self.model_optimizer.step()

    def update(self) -> Dict[str, float]:
        """Update SAC with world model augmentation - using correct log_prob."""
        self.total_steps += 1

        if self.replay_buffer.size < self.batch_size:
            return {}

        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        obs2 = torch.FloatTensor(batch["obs2"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch["done"]).unsqueeze(1).to(self.device)

        # Augment with imagined transitions from world model
        if self.total_steps > 5000:
            with torch.no_grad():
                imag_next_obs, imag_rew, imag_done = self.world_model(obs, act)
                mix_ratio = 0.5
                obs2 = obs2 * (1 - mix_ratio) + imag_next_obs * mix_ratio
                rew = rew * (1 - mix_ratio) + imag_rew * mix_ratio
                done = done * (1 - mix_ratio) + imag_done * mix_ratio

        # SAC update with correct log_prob
        q1_pred = self.q1(torch.cat([obs, act], dim=-1))
        q2_pred = self.q2(torch.cat([obs, act], dim=-1))

        with torch.no_grad():
            next_action, next_log_prob = self.policy(obs2, deterministic=False)

            next_q1 = self.q1_target(torch.cat([obs2, next_action], dim=-1))
            next_q2 = self.q2_target(torch.cat([obs2, next_action], dim=-1))
            min_next_q = torch.min(next_q1, next_q2)
            q_backup = rew + self.gamma * (1 - done) * (min_next_q - self.alpha * next_log_prob)

        q1_loss = F.mse_loss(q1_pred, q_backup)
        q2_loss = F.mse_loss(q2_pred, q_backup)

        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()

        # Policy update with correct log_prob
        curr_action, curr_log_prob = self.policy(obs, deterministic=False)

        q1_pi = self.q1(torch.cat([obs, curr_action], dim=-1))
        q2_pi = self.q2(torch.cat([obs, curr_action], dim=-1))
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * curr_log_prob - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_networks()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "policy": self.policy.state_dict(),
            "world_model": self.world_model.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])

    def close(self):
        """Cleanup resources."""
        pass
