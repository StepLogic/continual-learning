"""
Dual-Learner: Principled Fast and Meta Knowledge Learners for Continual RL
ICLR 2026

Inspired by hippocampus-cortex interaction in human memory:
- Fast learner: Rapid knowledge transfer (hippocampus-like)
- Meta learner: Slow knowledge integration (cortex-like)

The two learners are coupled to perform complementary roles in continual learning.
Uses SquashedGaussianPolicy for correct log_prob calculation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Tuple
from .sac import SACAgent, MLP, ReplayBuffer, SquashedGaussianPolicy


class DualLearnerAgent(SACAgent):
    """
    SAC agent with dual-learner architecture.

    Maintains two sets of networks:
    - Fast learner: Adapts quickly to new tasks, prone to forgetting
    - Meta learner: Slowly integrates knowledge, maintains stability

    Uses a gating mechanism to combine outputs from both learners.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        fast_lr: float = 0.001,
        meta_lr: float = 0.0001,
        switch_threshold: float = 0.5,
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
        self.fast_lr = fast_lr
        self.meta_lr = meta_lr
        self.switch_threshold = switch_threshold

        # Replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        # Fast learner networks (rapid adaptation) - use SquashedGaussianPolicy
        self.fast_q1 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.fast_q2 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.fast_policy = SquashedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(self.device)

        # Meta learner networks (slow integration) - use SquashedGaussianPolicy
        self.meta_q1 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.meta_q2 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.meta_policy = SquashedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(self.device)

        # Target networks (copy from meta learner for stability)
        self.meta_q1_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.meta_q2_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self._copy_meta_targets()

        # Gating network (decides learner weighting)
        self.gate = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)

        # Optimizers
        self.fast_optimizer = Adam(
            list(self.fast_q1.parameters())
            + list(self.fast_q2.parameters())
            + list(self.fast_policy.parameters())
            + list(self.gate.parameters()),
            lr=fast_lr,
        )
        self.meta_optimizer = Adam(
            list(self.meta_q1.parameters())
            + list(self.meta_q2.parameters())
            + list(self.meta_policy.parameters()),
            lr=meta_lr,
        )

        self.total_steps = 0
        self.task_steps = 0
        self.performance_buffer = []

    def _copy_meta_targets(self):
        """Copy meta Q networks to target networks."""
        for p, p_target in zip(self.meta_q1.parameters(), self.meta_q1_target.parameters()):
            p_target.data.copy_(p.data)
        for p, p_target in zip(self.meta_q2.parameters(), self.meta_q2_target.parameters()):
            p_target.data.copy_(p.data)

    def _update_meta_targets(self):
        """Soft update of meta target networks."""
        for p, p_target in zip(self.meta_q1.parameters(), self.meta_q1_target.parameters()):
            p_target.data.lerp_(p.data, self.tau)
        for p, p_target in zip(self.meta_q2.parameters(), self.meta_q2_target.parameters()):
            p_target.data.lerp_(p.data, self.tau)

    def get_action(self, obs: np.ndarray, deterministic: bool = False, random: bool = False) -> np.ndarray:
        """Get action by combining fast and meta learner outputs."""
        if random:
            return np.random.uniform(-1, 1, size=self.act_dim).astype(np.float32)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Get gating weight
        gate_weight = self.gate(obs_tensor)

        # Get actions from both learners
        fast_action, fast_log_prob = self.fast_policy(obs_tensor, deterministic=deterministic)
        meta_action, meta_log_prob = self.meta_policy(obs_tensor, deterministic=deterministic)

        # Combine actions
        combined_action = gate_weight * fast_action + (1 - gate_weight) * meta_action

        return combined_action.detach().cpu().numpy()[0]

    def store_transition(
        self, obs: np.ndarray, act: np.ndarray,
        rew: float, next_obs: np.ndarray, done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.store(obs, act, rew, next_obs, done)

    def update(self) -> Dict[str, float]:
        """Update both fast and meta learners."""
        self.total_steps += 1
        self.task_steps += 1

        if self.replay_buffer.size < self.batch_size:
            return {}

        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        obs2 = torch.FloatTensor(batch["obs2"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch["done"]).unsqueeze(1).to(self.device)

        # Get gating weights
        gate_weight = self.gate(obs)

        # ===== Update Fast Learner =====
        fast_q1_pred = self.fast_q1(torch.cat([obs, act], dim=-1))
        fast_q2_pred = self.fast_q2(torch.cat([obs, act], dim=-1))

        with torch.no_grad():
            # Use SquashedGaussianPolicy for correct log_prob
            meta_next_action, meta_next_log_prob = self.meta_policy(obs2, deterministic=False)

            meta_next_q1 = self.meta_q1_target(torch.cat([obs2, meta_next_action], dim=-1))
            meta_next_q2 = self.meta_q2_target(torch.cat([obs2, meta_next_action], dim=-1))
            min_next_q = torch.min(meta_next_q1, meta_next_q2)
            q_backup = rew + self.gamma * (1 - done) * (min_next_q - self.alpha * meta_next_log_prob)

        fast_q1_loss = F.mse_loss(fast_q1_pred, q_backup)
        fast_q2_loss = F.mse_loss(fast_q2_pred, q_backup)

        self.fast_optimizer.zero_grad()
        (fast_q1_loss + fast_q2_loss).backward()
        self.fast_optimizer.step()

        # ===== Update Meta Learner =====
        meta_q1_pred = self.meta_q1(torch.cat([obs, act], dim=-1))
        meta_q2_pred = self.meta_q2(torch.cat([obs, act], dim=-1))

        meta_q1_loss = F.mse_loss(meta_q1_pred, q_backup)
        meta_q2_loss = F.mse_loss(meta_q2_pred, q_backup)

        self.meta_optimizer.zero_grad()
        (meta_q1_loss + meta_q2_loss).backward()
        self.meta_optimizer.step()

        # Update meta target networks
        self._update_meta_targets()

        # Track performance for gating adjustment
        self.performance_buffer.append(-fast_q1_loss.item())
        if len(self.performance_buffer) > 100:
            self.performance_buffer.pop(0)

        # Adjust gating based on performance
        if self.task_steps % 100 == 0 and len(self.performance_buffer) > 10:
            avg_perf = np.mean(self.performance_buffer)
            if avg_perf < -self.switch_threshold:
                # Fast learner struggling, rely more on meta
                with torch.no_grad():
                    for param in self.gate.parameters():
                        param.mul_(0.95)

        return {
            "fast_q1_loss": fast_q1_loss.item(),
            "fast_q2_loss": fast_q2_loss.item(),
            "meta_q1_loss": meta_q1_loss.item(),
            "meta_q2_loss": meta_q2_loss.item(),
            "gate_weight": gate_weight.mean().item(),
        }

    def reset_task(self):
        """Called when switching to a new task."""
        self.task_steps = 0
        self.performance_buffer = []

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "fast_q1": self.fast_q1.state_dict(),
            "fast_q2": self.fast_q2.state_dict(),
            "fast_policy": self.fast_policy.state_dict(),
            "meta_q1": self.meta_q1.state_dict(),
            "meta_q2": self.meta_q2.state_dict(),
            "meta_policy": self.meta_policy.state_dict(),
            "gate": self.gate.state_dict(),
            "fast_optimizer": self.fast_optimizer.state_dict(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.fast_q1.load_state_dict(checkpoint["fast_q1"])
        self.fast_q2.load_state_dict(checkpoint["fast_q2"])
        self.fast_policy.load_state_dict(checkpoint["fast_policy"])
        self.meta_q1.load_state_dict(checkpoint["meta_q1"])
        self.meta_q2.load_state_dict(checkpoint["meta_q2"])
        self.meta_policy.load_state_dict(checkpoint["meta_policy"])
        self.gate.load_state_dict(checkpoint["gate"])
        self.fast_optimizer.load_state_dict(checkpoint["fast_optimizer"])
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])

    def close(self):
        """Cleanup resources."""
        pass
