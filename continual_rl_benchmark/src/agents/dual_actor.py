"""
Dual Actor Networks for Continual Reinforcement Learning
ICLR 2025

Addresses negative transfer by using two actor networks:
1. Current actor: Periodically resets to learn the current task
2. Distilled actor: Accumulates knowledge from all tasks via behavior cloning

The dual architecture balances plasticity (current actor) and stability (distilled actor).
Uses SquashedGaussianPolicy for correct log_prob calculation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List
from .sac import SACAgent, MLP, ReplayBuffer, SquashedGaussianPolicy


class DualActorAgent(SACAgent):
    """
    SAC agent with dual actor architecture.

    Uses two policies:
    - current_actor: Learns current task, periodically resets
    - distilled_actor: Distills knowledge from all tasks via BC
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        reset_interval: int = 50000,
        distill_alpha: float = 0.5,
        bc_buffer_size: int = 10000,
        buffer_size: int = 10000,
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
        self.reset_interval = reset_interval
        self.distill_alpha = distill_alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        # Behavior cloning buffer (stores best trajectories)
        self.bc_buffer_obs = np.zeros((bc_buffer_size, obs_dim), dtype=np.float32)
        self.bc_buffer_act = np.zeros((bc_buffer_size, act_dim), dtype=np.float32)
        self.bc_ptr, self.bc_size = 0, 0
        self.bc_max_size = bc_buffer_size

        # Q-networks (shared)
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q1_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)
        self.q2_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(self.device)

        # Current actor (task-specific, resets periodically) - use SquashedGaussianPolicy
        self.current_actor = SquashedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(self.device)

        # Distilled actor (accumulates all knowledge) - use SquashedGaussianPolicy
        self.distilled_actor = SquashedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(self.device)

        self._copy_target_networks()

        # Optimizers
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=learning_rate
        )
        self.current_actor_optimizer = Adam(self.current_actor.parameters(), lr=learning_rate)
        self.distilled_actor_optimizer = Adam(self.distilled_actor.parameters(), lr=learning_rate * 0.5)

        self.total_steps = 0
        self.steps_since_reset = 0

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
        """Get action by combining current and distilled actors."""
        if random:
            return np.random.uniform(-1, 1, size=self.act_dim).astype(np.float32)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Get actions from both actors
        curr_action, _ = self.current_actor(obs_tensor, deterministic=deterministic)
        dist_action, _ = self.distilled_actor(obs_tensor, deterministic=deterministic)

        # Combine: weighted average based on task progress
        progress = min(1.0, self.steps_since_reset / (self.reset_interval * 0.8))
        curr_weight = 1.0 - progress * 0.5
        dist_weight = 1.0 - curr_weight

        combined_action = curr_weight * curr_action + dist_weight * dist_action
        return combined_action.detach().cpu().numpy()[0]

    def store_transition(
        self, obs: np.ndarray, act: np.ndarray,
        rew: float, next_obs: np.ndarray, done: bool
    ):
        """Store transition in both buffers."""
        self.replay_buffer.store(obs, act, rew, next_obs, done)

        if rew > 0:
            self.bc_buffer_obs[self.bc_ptr] = obs
            self.bc_buffer_act[self.bc_ptr] = act
            self.bc_ptr = (self.bc_ptr + 1) % self.bc_max_size
            self.bc_size = min(self.bc_size + 1, self.bc_max_size)

    def _reset_current_actor(self):
        """Reset current actor weights to prevent overfitting."""
        def reset_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.current_actor.apply(reset_layer)
        self.steps_since_reset = 0

    def update(self) -> Dict[str, float]:
        """Update with dual actor training - using correct SAC log_prob."""
        self.total_steps += 1
        self.steps_since_reset += 1

        if self.steps_since_reset >= self.reset_interval:
            self._reset_current_actor()

        if self.replay_buffer.size < self.batch_size:
            return {}

        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        obs2 = torch.FloatTensor(batch["obs2"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch["done"]).unsqueeze(1).to(self.device)

        # Q-network update
        q1_pred = self.q1(torch.cat([obs, act], dim=-1))
        q2_pred = self.q2(torch.cat([obs, act], dim=-1))

        with torch.no_grad():
            # Use distilled actor with correct log_prob
            dist_next_action, dist_next_log_prob = self.distilled_actor(obs2, deterministic=False)

            next_q1 = self.q1_target(torch.cat([obs2, dist_next_action], dim=-1))
            next_q2 = self.q2_target(torch.cat([obs2, dist_next_action], dim=-1))
            min_next_q = torch.min(next_q1, next_q2)
            q_backup = rew + self.gamma * (1 - done) * (min_next_q - self.alpha * dist_next_log_prob)

        q1_loss = F.mse_loss(q1_pred, q_backup)
        q2_loss = F.mse_loss(q2_pred, q_backup)

        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()

        # Current actor update (standard SAC with correct log_prob)
        curr_action, curr_log_prob = self.current_actor(obs, deterministic=False)

        curr_q1 = self.q1(torch.cat([obs, curr_action], dim=-1))
        curr_q2 = self.q2(torch.cat([obs, curr_action], dim=-1))
        curr_min_q = torch.min(curr_q1, curr_q2)

        current_actor_loss = (self.alpha * curr_log_prob - curr_min_q).mean()

        self.current_actor_optimizer.zero_grad()
        current_actor_loss.backward()
        self.current_actor_optimizer.step()

        # Distilled actor update
        dist_action, dist_log_prob = self.distilled_actor(obs, deterministic=False)

        dist_q1 = self.q1(torch.cat([obs, dist_action], dim=-1))
        dist_q2 = self.q2(torch.cat([obs, dist_action], dim=-1))
        dist_min_q = torch.min(dist_q1, dist_q2)

        distilled_sac_loss = (self.alpha * dist_log_prob - dist_min_q).mean()

        # Distillation loss: match current actor's actions
        with torch.no_grad():
            target_action, _ = self.current_actor(obs, deterministic=True)
        distill_loss = F.mse_loss(dist_action, target_action)

        # Behavior cloning loss
        if self.bc_size > 0:
            bc_size = min(self.batch_size, self.bc_size)
            bc_idx = np.random.randint(0, self.bc_size, size=bc_size)
            bc_obs = torch.FloatTensor(self.bc_buffer_obs[bc_idx]).to(self.device)
            bc_act = torch.FloatTensor(self.bc_buffer_act[bc_idx]).to(self.device)

            bc_mu, _ = self.distilled_actor(bc_obs, deterministic=True)
            bc_loss = F.mse_loss(bc_mu, bc_act)
        else:
            bc_loss = torch.tensor(0.0, device=self.device)

        total_distill_loss = distilled_sac_loss + self.distill_alpha * distill_loss + bc_loss

        self.distilled_actor_optimizer.zero_grad()
        total_distill_loss.backward()
        self.distilled_actor_optimizer.step()

        self._update_target_networks()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "current_actor_loss": current_actor_loss.item(),
            "distilled_actor_loss": total_distill_loss.item(),
            "distill_loss": distill_loss.item(),
            "bc_loss": bc_loss.item(),
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "current_actor": self.current_actor.state_dict(),
            "distilled_actor": self.distilled_actor.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "current_actor_optimizer": self.current_actor_optimizer.state_dict(),
            "distilled_actor_optimizer": self.distilled_actor_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.current_actor.load_state_dict(checkpoint["current_actor"])
        self.distilled_actor.load_state_dict(checkpoint["distilled_actor"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.current_actor_optimizer.load_state_dict(checkpoint["current_actor_optimizer"])
        self.distilled_actor_optimizer.load_state_dict(checkpoint["distilled_actor_optimizer"])

    def close(self):
        """Cleanup resources."""
        pass
