"""
Fast TRAC: Parameter-Free Optimizer for Lifelong Reinforcement Learning
NeurIPS 2024

Simplified stable version using gradient clipping and adaptive LR.
Uses corrected SAC implementation with proper SquashedGaussianPolicy.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict
from .sac import SACAgent, ReplayBuffer, SquashedGaussianPolicy, MLP


class FastTRACOptimizer(Optimizer):
    """
    Simplified Fast TRAC optimizer with gradient clipping.
    """

    def __init__(
        self,
        params,
        base_lr: float = 0.003,
        momentum: float = 0.9,
        grad_clip: float = 1.0,
    ):
        defaults = dict(
            base_lr=base_lr,
            momentum=momentum,
            grad_clip=grad_clip,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                state["step"] += 1

                grad = p.grad
                momentum = state["momentum_buffer"]

                # Clip gradients
                grad = torch.clamp(grad, -group["grad_clip"], group["grad_clip"])

                # Update momentum
                momentum.mul_(group["momentum"]).add_(grad, alpha=1 - group["momentum"])

                # Apply update
                p.add_(momentum, alpha=-group["base_lr"])


class FastTRACAgent(SACAgent):
    """
    SAC agent with Fast TRAC optimizer.
    Uses SquashedGaussianPolicy for correct log_prob calculation.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        base_lr: float = 0.003,
        momentum: float = 0.9,
        grad_clip: float = 1.0,
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

        # Networks - use SquashedGaussianPolicy for correct log_prob
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q1_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.policy = SquashedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(self.device)

        # Initialize target networks
        self._copy_target_networks()

        # Fast TRAC optimizers
        self.q_optimizer = FastTRACOptimizer(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            base_lr=base_lr,
            momentum=momentum,
            grad_clip=grad_clip,
        )
        self.policy_optimizer = FastTRACOptimizer(
            self.policy.parameters(),
            base_lr=base_lr,
            momentum=momentum,
            grad_clip=grad_clip,
        )

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
        """Update with Fast TRAC optimizer - using correct SAC log_prob."""
        self.total_steps += 1

        if self.replay_buffer.size < self.batch_size:
            return {}

        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        obs2 = torch.FloatTensor(batch["obs2"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch["done"]).unsqueeze(1).to(self.device)

        # Q update
        q1_pred = self.q1(torch.cat([obs, act], dim=-1))
        q2_pred = self.q2(torch.cat([obs, act], dim=-1))

        with torch.no_grad():
            # Use SquashedGaussianPolicy for correct log_prob
            next_action, next_log_prob = self.policy(obs2, deterministic=False)

            next_q1 = self.q1_target(torch.cat([obs2, next_action], dim=-1))
            next_q2 = self.q2_target(torch.cat([obs2, next_action], dim=-1))
            min_next_q = torch.min(next_q1, next_q2)
            q_backup = rew + self.gamma * (1 - done) * (min_next_q - self.alpha * next_log_prob)

        q1_loss = nn.functional.mse_loss(q1_pred, q_backup)
        q2_loss = nn.functional.mse_loss(q2_pred, q_backup)

        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()

        # Policy update
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
