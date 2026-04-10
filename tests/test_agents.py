"""
Unit tests for agent implementations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents import (
    SACAgent,
    FastTRACAgent,
    ParsevalAgent,
    DualLearnerAgent,
    WorldModelAgent,
    DualActorAgent,
)


@pytest.fixture
def obs_dim():
    return 10


@pytest.fixture
def act_dim():
    return 4


@pytest.fixture
def dummy_transition(obs_dim, act_dim):
    return {
        "obs": np.random.randn(obs_dim).astype(np.float32),
        "act": np.random.randn(act_dim).astype(np.float32),
        "rew": 1.0,
        "next_obs": np.random.randn(obs_dim).astype(np.float32),
        "done": False,
    }


class TestSACAgent:
    def test_initialization(self, obs_dim, act_dim):
        agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)
        assert agent.replay_buffer is not None
        assert agent.q1 is not None
        assert agent.q2 is not None
        assert agent.policy is not None

    def test_get_action(self, obs_dim, act_dim):
        agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.get_action(obs)
        assert action.shape == (act_dim,)

    def test_store_and_update(self, obs_dim, act_dim, dummy_transition):
        agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, batch_size=32)

        # Store multiple transitions
        for _ in range(50):
            agent.store_transition(
                dummy_transition["obs"],
                dummy_transition["act"],
                dummy_transition["rew"],
                dummy_transition["next_obs"],
                dummy_transition["done"],
            )

        # Update should work
        metrics = agent.update()
        assert "q1_loss" in metrics or metrics == {}

    def test_deterministic_action(self, obs_dim, act_dim):
        agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action1 = agent.get_action(obs, deterministic=True)
        action2 = agent.get_action(obs, deterministic=True)
        assert np.allclose(action1, action2)


class TestFastTRACAgent:
    def test_initialization(self, obs_dim, act_dim):
        agent = FastTRACAgent(obs_dim=obs_dim, act_dim=act_dim)
        assert agent is not None

    def test_get_action(self, obs_dim, act_dim):
        agent = FastTRACAgent(obs_dim=obs_dim, act_dim=act_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.get_action(obs)
        assert action.shape == (act_dim,)


class TestParsevalAgent:
    def test_initialization(self, obs_dim, act_dim):
        agent = ParsevalAgent(obs_dim=obs_dim, act_dim=act_dim)
        assert agent is not None

    def test_orthogonality_loss(self, obs_dim, act_dim):
        agent = ParsevalAgent(obs_dim=obs_dim, act_dim=act_dim)
        loss = agent.q1.orthogonality_loss()
        assert loss >= 0


class TestDualLearnerAgent:
    def test_initialization(self, obs_dim, act_dim):
        agent = DualLearnerAgent(obs_dim=obs_dim, act_dim=act_dim)
        assert agent.fast_q1 is not None
        assert agent.meta_q1 is not None
        assert agent.gate is not None

    def test_get_action(self, obs_dim, act_dim):
        agent = DualLearnerAgent(obs_dim=obs_dim, act_dim=act_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.get_action(obs)
        assert action.shape == (act_dim,)


class TestWorldModelAgent:
    def test_initialization(self, obs_dim, act_dim):
        agent = WorldModelAgent(obs_dim=obs_dim, act_dim=act_dim)
        assert agent.world_model is not None
        assert agent.mpc_planner is not None

    def test_get_action(self, obs_dim, act_dim):
        agent = WorldModelAgent(obs_dim=obs_dim, act_dim=act_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.get_action(obs)
        assert action.shape == (act_dim,)


class TestDualActorAgent:
    def test_initialization(self, obs_dim, act_dim):
        agent = DualActorAgent(obs_dim=obs_dim, act_dim=act_dim)
        assert agent.current_actor is not None
        assert agent.distilled_actor is not None

    def test_get_action(self, obs_dim, act_dim):
        agent = DualActorAgent(obs_dim=obs_dim, act_dim=act_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.get_action(obs)
        assert action.shape == (act_dim,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
