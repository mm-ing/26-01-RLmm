import numpy as np
import torch

from card_pole_logic import (
    CartPoleEnv,
    DQNPolicy,
    DDQNPolicy,
    Agent,
)


def test_env_step_returns_values():
    env = CartPoleEnv(seed=0)
    state = env.reset()
    assert isinstance(state, np.ndarray)
    next_state, reward, done, _ = env.step(0)
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()


def test_dqn_learn_updates_weights():
    torch.manual_seed(0)
    policy = DQNPolicy(
        state_dim=4,
        action_dim=2,
        alpha=0.01,
        gamma=0.9,
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=0.0,
        hidden_layers=1,
        hidden_units=8,
        batch_size=1,
        replay_size=10,
        activation="ReLU",
    )
    state = np.zeros(4, dtype=np.float32)
    next_state = np.ones(4, dtype=np.float32)
    policy.store(state, 0, 1.0, next_state, False)
    before = next(policy.q_net.parameters()).detach().clone()
    policy.learn()
    after = next(policy.q_net.parameters()).detach().clone()
    assert not torch.equal(before, after)


def test_ddqn_learn_updates_weights():
    torch.manual_seed(0)
    policy = DDQNPolicy(
        state_dim=4,
        action_dim=2,
        alpha=0.01,
        gamma=0.9,
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=0.0,
        hidden_layers=1,
        hidden_units=8,
        batch_size=1,
        replay_size=10,
        activation="ReLU",
    )
    state = np.zeros(4, dtype=np.float32)
    next_state = np.ones(4, dtype=np.float32)
    policy.store(state, 0, 1.0, next_state, False)
    before = next(policy.q_net.parameters()).detach().clone()
    policy.learn()
    after = next(policy.q_net.parameters()).detach().clone()
    assert not torch.equal(before, after)


def test_agent_runs_episode():
    env = CartPoleEnv(seed=0)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    policy = DQNPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        alpha=0.01,
        gamma=0.9,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.05,
        hidden_layers=1,
        hidden_units=8,
        batch_size=1,
        replay_size=10,
        activation="ReLU",
    )
    agent = Agent(env, policy)
    total = agent.run_episode(0, max_steps=10, step_delay=0.0)
    assert isinstance(total, float)
    env.close()
