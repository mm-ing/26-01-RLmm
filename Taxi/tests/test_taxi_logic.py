from taxi_logic import (
    TaxiEnv,
    QLearningPolicy,
    SarsaPolicy,
    ExpectedSarsaPolicy,
    Agent,
)


def test_env_step_returns_values():
    env = TaxiEnv(is_raining=False, seed=0)
    state = env.reset()
    assert isinstance(state, int)
    next_state, reward, done, _ = env.step(0)
    assert isinstance(next_state, int)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()


def test_qlearning_update_changes_q():
    policy = QLearningPolicy(n_states=8, n_actions=2, alpha=0.5, gamma=0.9, eps_start=0.0, eps_end=0.0, total_episodes=1)
    policy.update(0, 1, 1.0, 0, True)
    assert policy.q[0, 1] > 0.0


def test_sarsa_update_changes_q():
    policy = SarsaPolicy(n_states=8, n_actions=2, alpha=0.5, gamma=0.9, eps_start=0.0, eps_end=0.0, total_episodes=1)
    policy.update(0, 0, 1.0, 1, True, next_action=0)
    assert policy.q[0, 0] > 0.0


def test_expected_sarsa_update_changes_q():
    policy = ExpectedSarsaPolicy(n_states=8, n_actions=2, alpha=0.5, gamma=0.9, eps_start=0.0, eps_end=0.0, total_episodes=1)
    policy.update(0, 0, 1.0, 1, True)
    assert policy.q[0, 0] > 0.0


def test_agent_runs_episode():
    env = TaxiEnv(is_raining=False, seed=0)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = QLearningPolicy(n_states, n_actions, alpha=0.2, gamma=0.9, eps_start=1.0, eps_end=0.05, total_episodes=1)
    agent = Agent(env, policy)
    total = agent.run_episode(0, 1, max_steps=20, step_delay=0.0)
    assert isinstance(total, float)
    env.close()
