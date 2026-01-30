import numpy as np
from gymnasium import spaces

from mountain_car_logic import (
    MountainCarEnv,
    DQNPolicy,
    PrioritizedDDQNPolicy,
    RainbowPolicy,
    Agent,
)


def test_env_step_returns_values():
    env = MountainCarEnv(seed=0, goal_velocity=0.0)
    state = env.reset()
    assert state.shape == (2,)
    action = env.action_from_index(0)
    next_state, reward, done, _ = env.step(action)
    assert next_state.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()


def _fill_policy(policy):
    state = np.array([0.0, 0.0], dtype=np.float32)
    next_state = np.array([0.01, 0.0], dtype=np.float32)
    for _ in range(policy.batch_size):
        policy.store(state, 0, 1.0, next_state, False)
    policy.learn()


def test_dqn_learn_runs():
    policy = DQNPolicy(
        state_dim=2,
        action_dim=3,
        alpha=0.001,
        gamma=0.99,
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=0.0,
        hidden_layers=1,
        hidden_units=16,
        batch_size=4,
        replay_size=100,
        activation="ReLU",
        warmup_steps=1,
        normalize_states=False,
        reward_scale=1.0,
        grad_clip=10.0,
    )
    _fill_policy(policy)


def test_prioritized_ddqn_learn_runs():
    policy = PrioritizedDDQNPolicy(
        state_dim=2,
        action_dim=3,
        alpha=0.001,
        gamma=0.99,
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=0.0,
        hidden_layers=1,
        hidden_units=16,
        batch_size=4,
        replay_size=100,
        activation="ReLU",
        warmup_steps=1,
        normalize_states=False,
        reward_scale=1.0,
        grad_clip=10.0,
    )
    _fill_policy(policy)


def test_rainbow_learn_runs():
    policy = RainbowPolicy(
        state_dim=2,
        action_dim=3,
        alpha=0.001,
        gamma=0.99,
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=0.0,
        hidden_layers=1,
        hidden_units=16,
        batch_size=4,
        replay_size=100,
        activation="ReLU",
        warmup_steps=1,
        normalize_states=False,
        reward_scale=1.0,
        grad_clip=10.0,
        atoms=11,
        v_min=-10.0,
        v_max=0.0,
    )
    _fill_policy(policy)


def test_agent_runs_episode():
    env = MountainCarEnv(seed=0, goal_velocity=0.0)
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    state_dim = int(obs_shape[0])
    assert isinstance(env.action_space, spaces.Box)
    action_dim = env.action_bins
    policy = DQNPolicy(state_dim, action_dim, alpha=0.001, gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=0.0, hidden_layers=1, hidden_units=16, batch_size=4, replay_size=100, activation="ReLU", warmup_steps=1, normalize_states=False, reward_scale=1.0, grad_clip=10.0)
    agent = Agent(env, policy)
    total = agent.run_episode(0, max_steps=10, step_delay=0.0)
    assert isinstance(total, float)
    env.close()
