"""
Soft Actor-Critic (SAC) implementation.
Base agent for all continual RL methods.

Corrected implementation following stable-baselines3 and SpinningUp.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Tuple, Optional
from collections import deque


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs: np.ndarray, act: np.ndarray,
              rew: float, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )


class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianPolicy(nn.Module):
    """
    Squashed Gaussian policy following stable-baselines3.

    Outputs mean and log_std separately for proper log_prob calculation.
    Supports optional orthogonality regularization for Parseval agent.
    """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, use_orthogonality: bool = False):
        super().__init__()
        self.act_dim = act_dim
        self.use_orthogonality = use_orthogonality

        # Shared feature extractor
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.feature_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
        )

        # Mean and log_std heads
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action and log_prob.

        Returns:
            action: Squashed action (tanh output)
            log_prob: Log probability of the action under the squashed Gaussian
        """
        features = self.feature_net(obs)
        mu = self.mu_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Sample from Gaussian (before squashing)
        if deterministic:
            z = mu
        else:
            z = mu + std * torch.randn_like(mu)

        # Squash the action
        action = torch.tanh(z)

        # Compute log_prob with squashing correction
        # log_prob = log(N(z; mu, std)) - sum(log(1 - a^2 + eps))
        # where a = tanh(z)
        # Epsilon for numerical stability in squashing correction
        eps = 1e-6

        # Gaussian log probability
        log_prob_gaussian = -0.5 * (((z - mu) / std) ** 2 + 2 * log_std + np.log(2 * np.pi)).sum(dim=-1, keepdim=True)

        # Squashing correction (Jacobian term) - clip to avoid NaN
        action_clipped = torch.clamp(action, -1 + eps, 1 - eps)
        log_prob_correction = torch.log(1 - action_clipped ** 2 + eps).sum(dim=-1, keepdim=True)
        log_prob = log_prob_gaussian - log_prob_correction

        return action, log_prob

    def get_mu_log_std(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mu and log_std for sampling."""
        features = self.feature_net(obs)
        mu = self.mu_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def orthogonality_loss(self, lambda_reg: float = 0.1) -> torch.Tensor:
        """
        Compute orthogonality regularization loss for Parseval agent.

        For each weight matrix W, penalize deviation from orthogonality:
        L_orth = ||W^T W - I||_F^2
        """
        if not self.use_orthogonality:
            return torch.tensor(0.0, device=self.mu_head.weight.device)

        orth_loss = 0.0
        for layer in [self.fc1, self.fc2]:
            W = layer.weight
            WTW = W @ W.t()
            I = torch.eye(WTW.shape[0], device=W.device)
            orth_loss = orth_loss + torch.sum((WTW - I) ** 2)

        return lambda_reg * orth_loss


class SACAgent:
    """
    Soft Actor-Critic agent.

    Uses twin Q-functions for stability.
    Implementation follows stable-baselines3 and SpinningUp.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
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

        # Replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        # Networks - use proper SquashedGaussianPolicy
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q1_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.policy = SquashedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(self.device)

        # Initialize target networks
        self._copy_target_networks()

        # Optimizers
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=learning_rate
        )
        self.policy_optimizer = Adam(self.policy.parameters(), lr=learning_rate)

        # Training stats
        self.total_steps = 0

    def _copy_target_networks(self):
        """Copy Q networks to target networks."""
        for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_target.data.copy_(p.data)
        for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_target.data.copy_(p.data)

    def _update_target_networks(self):
        """Soft update of target networks."""
        for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_target.data.lerp_(p.data, self.tau)
        for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_target.data.lerp_(p.data, self.tau)

    def get_action(self, obs: np.ndarray, deterministic: bool = False, random: bool = False) -> np.ndarray:
        """
        Sample action from policy.

        Args:
            obs: Observation
            deterministic: If True, use mean action (for evaluation)
            random: If True, return uniform random action (for exploration warmup)
        """
        if random:
            # Return uniform random action in [-1, 1]
            return np.random.uniform(-1, 1, size=self.act_dim).astype(np.float32)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, _ = self.policy(obs_tensor, deterministic=deterministic)
        return action.detach().cpu().numpy()[0]

    def store_transition(
        self, obs: np.ndarray, act: np.ndarray,
        rew: float, next_obs: np.ndarray, done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.store(obs, act, rew, next_obs, done)

    def update(self) -> Dict[str, float]:
        """
        Perform one update step.

        Correctly implements SAC with:
        - Twin Q-functions
        - Squashed Gaussian policy
        - Proper log_prob calculation with squashing correction
        """
        self.total_steps += 1

        if self.replay_buffer.size < self.batch_size:
            return {}

        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        obs2 = torch.FloatTensor(batch["obs2"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch["done"]).unsqueeze(1).to(self.device)

        # ============ Update Q-functions ============
        q1_pred = self.q1(torch.cat([obs, act], dim=-1))
        q2_pred = self.q2(torch.cat([obs, act], dim=-1))

        with torch.no_grad():
            # Sample next action from policy (with proper squashed Gaussian)
            next_action, next_log_prob = self.policy(obs2, deterministic=False)

            # Target Q-values with entropy regularization
            next_q1 = self.q1_target(torch.cat([obs2, next_action], dim=-1))
            next_q2 = self.q2_target(torch.cat([obs2, next_action], dim=-1))
            min_next_q = torch.min(next_q1, next_q2)

            # Bellman backup: r + gamma * (min(Q1, Q2) - alpha * log_prob)
            q_backup = rew + self.gamma * (1.0 - done) * (min_next_q - self.alpha * next_log_prob)

        # Q-function losses (0.5 factor for proper gradient scaling, matching stable-baselines3)
        q1_loss = 0.5 * F.mse_loss(q1_pred, q_backup)
        q2_loss = 0.5 * F.mse_loss(q2_pred, q_backup)

        # Check for NaN/Inf
        if torch.isnan(q1_loss) or torch.isinf(q1_loss):
            return {"error": "Q1 loss is NaN/Inf"}
        if torch.isnan(q2_loss) or torch.isinf(q2_loss):
            return {"error": "Q2 loss is NaN/Inf"}

        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()

        # ============ Update Policy ============
        # Sample current action and log_prob
        curr_action, curr_log_prob = self.policy(obs, deterministic=False)

        # Get Q-values for the sampled actions
        q1_pi = self.q1(torch.cat([obs, curr_action], dim=-1))
        q2_pi = self.q2(torch.cat([obs, curr_action], dim=-1))
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Policy loss: maximize expected Q-value minus entropy bonus
        # Equivalent to minimizing: alpha * log_prob - min_q_pi
        policy_loss = (self.alpha * curr_log_prob - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ============ Update Target Networks ============
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
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])

    def close(self):
        """Cleanup resources."""
        pass
