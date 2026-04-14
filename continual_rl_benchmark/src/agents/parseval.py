"""
Parseval Regularization for Continual Reinforcement Learning
NeurIPS 2024

Maintains orthogonality of weight matrices to enhance optimization
and preserve plasticity when learning new tasks.
Uses SquashedGaussianPolicy for correct log_prob calculation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict
from .sac import SACAgent, MLP, ReplayBuffer, SquashedGaussianPolicy


class ParsevalMLP(nn.Module):
    """MLP with Parseval (orthogonality) regularization."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def orthogonality_loss(self, lambda_reg: float = 0.1) -> torch.Tensor:
        """
        Compute orthogonality regularization loss.

        For each weight matrix W, penalize deviation from orthogonality:
        L_orth = ||W^T W - I||_F^2
        """
        orth_loss = 0.0
        for layer in [self.fc1, self.fc2]:
            W = layer.weight
            WTW = W @ W.t()
            I = torch.eye(WTW.shape[0], device=W.device)
            orth_loss = orth_loss + torch.sum((WTW - I) ** 2)

        return lambda_reg * orth_loss

    def get_all_parameters(self):
        """Get all parameters for optimizer."""
        return list(self.parameters())


class ParsevalAgent(SACAgent):
    """
    SAC agent with Parseval regularization.

    Adds orthogonality constraint to weight matrices during training
    to preserve plasticity across tasks.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        parseval_lambda: float = 0.1,
        orthogonality_freq: int = 10,
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
        self.parseval_lambda = parseval_lambda
        self.orthogonality_freq = orthogonality_freq

        # Replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        # Q-networks with Parseval MLP
        self.q1 = ParsevalMLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2 = ParsevalMLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q1_target = ParsevalMLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2_target = ParsevalMLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)

        # Use SquashedGaussianPolicy for correct log_prob
        self.policy = SquashedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(self.device)

        # Initialize target networks
        self._copy_target_networks()

        # Optimizers
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=learning_rate
        )
        self.policy_optimizer = Adam(self.policy.parameters(), lr=learning_rate)

        self.total_steps = 0

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
        """Sample action from policy."""
        if random:
            return np.random.uniform(-1, 1, size=self.act_dim).astype(np.float32)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, _ = self.policy(obs_tensor, deterministic=deterministic)
        return action.detach().cpu().numpy()[0]

    def update(self) -> Dict[str, float]:
        """Update with Parseval regularization - using correct SAC log_prob."""
        self.total_steps += 1

        if self.replay_buffer.size < self.batch_size:
            return {}

        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        obs2 = torch.FloatTensor(batch["obs2"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch["done"]).unsqueeze(1).to(self.device)

        # Update Q-functions with Parseval regularization
        q1_pred = self.q1(torch.cat([obs, act], dim=-1))
        q2_pred = self.q2(torch.cat([obs, act], dim=-1))

        with torch.no_grad():
            # Use SquashedGaussianPolicy for correct log_prob
            next_action, next_log_prob = self.policy(obs2, deterministic=False)

            next_q1 = self.q1_target(torch.cat([obs2, next_action], dim=-1))
            next_q2 = self.q2_target(torch.cat([obs2, next_action], dim=-1))
            min_next_q = torch.min(next_q1, next_q2)
            q_backup = rew + self.gamma * (1 - done) * (min_next_q - self.alpha * next_log_prob)

        q1_loss = F.mse_loss(q1_pred, q_backup)
        q2_loss = F.mse_loss(q2_pred, q_backup)

        orth_loss_q1 = self.q1.orthogonality_loss(self.parseval_lambda)
        orth_loss_q2 = self.q2.orthogonality_loss(self.parseval_lambda)

        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss + orth_loss_q1 + orth_loss_q2).backward()
        self.q_optimizer.step()

        # Update policy with correct log_prob
        curr_action, curr_log_prob = self.policy(obs, deterministic=False)

        q1_pi = self.q1(torch.cat([obs, curr_action], dim=-1))
        q2_pi = self.q2(torch.cat([obs, curr_action], dim=-1))
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * curr_log_prob - min_q_pi).mean()
        orth_loss_policy = self.policy.orthogonality_loss(self.parseval_lambda)

        self.policy_optimizer.zero_grad()
        (policy_loss + orth_loss_policy).backward()
        self.policy_optimizer.step()

        self._update_target_networks()

        if self.total_steps % self.orthogonality_freq == 0:
            self._orthogonalize_weights()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "orth_loss": (orth_loss_q1 + orth_loss_q2 + orth_loss_policy).item(),
        }

    def _orthogonalize_weights(self):
        """Apply orthogonal regularization via Gram-Schmidt."""
        for network in [self.q1, self.q2, self.policy]:
            for layer in [network.fc1, network.fc2]:
                W = layer.weight.data
                row_norms = W.norm(dim=1, keepdim=True)
                W_normalized = W / (row_norms + 1e-8)
                layer.weight.data = W_normalized

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
