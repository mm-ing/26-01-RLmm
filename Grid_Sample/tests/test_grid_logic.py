from grid_logic import GridWorld, QlearningPolicies, Agent


def test_blocked_cell_stays_in_place():
    env = GridWorld(width=3, height=3, start=(0, 0), goal=(2, 2), blocked=[(1, 0)])
    env.reset()
    nxt, _, _, _ = env.step(3)  # right into blocked
    assert nxt == (0, 0)


def test_reaches_goal_reward_zero():
    env = GridWorld(width=2, height=1, start=(0, 0), goal=(1, 0))
    env.reset()
    nxt, reward, done, _ = env.step(3)
    assert nxt == (1, 0)
    assert reward == 0.0
    assert done is True


def test_qlearning_updates():
    env = GridWorld(width=2, height=1, start=(0, 0), goal=(1, 0))
    policy = QlearningPolicies(alpha=0.5, gamma=0.9)
    agent = Agent(env, policy)
    total, steps = agent.run_episode(max_steps=2)
    assert steps >= 1
    assert isinstance(total, float)
