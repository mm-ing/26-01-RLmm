import unittest
import numpy as np
from lunar_lander_c_logic import (
    LunarLanderCEnv,
    RainbowPolicy,
    A2CPolicy,
    TRPOPolicy,
    PPOPolicy,
    Agent,
    StepResult,
)


class TestLunarLanderCEnv(unittest.TestCase):
    def test_env_creation_continuous(self):
        env = LunarLanderCEnv(continuous=True)
        self.assertTrue(env.continuous)
        self.assertEqual(env.action_dim, 2)
        env.close()

    def test_env_reset(self):
        env = LunarLanderCEnv(continuous=True)
        state = env.reset()
        self.assertEqual(state.shape, (8,))
        self.assertEqual(state.dtype, np.float32)
        env.close()

    def test_env_step(self):
        env = LunarLanderCEnv(continuous=True)
        env.reset()
        action = np.array([0.5, 0.5], dtype=np.float32)
        next_state, reward, done, info = env.step(action)
        self.assertEqual(next_state.shape, (8,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        env.close()


class TestRainbowPolicy(unittest.TestCase):
    def test_policy_creation(self):
        policy = RainbowPolicy(state_dim=8, action_dim=2)
        self.assertEqual(policy.state_dim, 8)
        self.assertEqual(policy.action_dim, 2)
        self.assertEqual(policy.steps, 0)

    def test_select_action(self):
        policy = RainbowPolicy(state_dim=8, action_dim=2)
        state = np.random.randn(8).astype(np.float32)
        action = policy.select_action(state)
        self.assertEqual(action.shape, (2,))
        self.assertEqual(action.dtype, np.float32)

    def test_update(self):
        policy = RainbowPolicy(state_dim=8, action_dim=2, warmup_steps=10)
        state = np.random.randn(8).astype(np.float32)
        action = np.array([0.5, 0.5], dtype=np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        reward = 1.0
        done = False
        
        step_result = StepResult(state, action, reward, next_state, done)
        policy.update(step_result)
        self.assertEqual(policy.steps, 1)

    def test_reset(self):
        policy = RainbowPolicy(state_dim=8, action_dim=2)
        policy.steps = 100
        policy.reset()
        self.assertEqual(policy.steps, 0)


class TestA2CPolicy(unittest.TestCase):
    def test_policy_creation(self):
        policy = A2CPolicy(state_dim=8, action_dim=2)
        self.assertEqual(policy.state_dim, 8)
        self.assertEqual(policy.action_dim, 2)

    def test_select_action(self):
        policy = A2CPolicy(state_dim=8, action_dim=2)
        state = np.random.randn(8).astype(np.float32)
        action = policy.select_action(state)
        self.assertEqual(action.shape, (2,))

    def test_update(self):
        policy = A2CPolicy(state_dim=8, action_dim=2, n_steps=5)
        for _ in range(5):
            state = np.random.randn(8).astype(np.float32)
            action = np.array([0.5, 0.5], dtype=np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            reward = 1.0
            done = False
            
            step_result = StepResult(state, action, reward, next_state, done)
            policy.update(step_result)
        
        self.assertEqual(policy.steps, 5)


class TestTRPOPolicy(unittest.TestCase):
    def test_policy_creation(self):
        policy = TRPOPolicy(state_dim=8, action_dim=2)
        self.assertEqual(policy.state_dim, 8)
        self.assertEqual(policy.action_dim, 2)

    def test_select_action(self):
        policy = TRPOPolicy(state_dim=8, action_dim=2)
        state = np.random.randn(8).astype(np.float32)
        action = policy.select_action(state)
        self.assertEqual(action.shape, (2,))


class TestPPOPolicy(unittest.TestCase):
    def test_policy_creation(self):
        policy = PPOPolicy(state_dim=8, action_dim=2)
        self.assertEqual(policy.state_dim, 8)
        self.assertEqual(policy.action_dim, 2)

    def test_select_action(self):
        policy = PPOPolicy(state_dim=8, action_dim=2)
        state = np.random.randn(8).astype(np.float32)
        action = policy.select_action(state)
        self.assertEqual(action.shape, (2,))

    def test_update(self):
        policy = PPOPolicy(state_dim=8, action_dim=2, n_steps=10)
        for _ in range(10):
            state = np.random.randn(8).astype(np.float32)
            action = np.array([0.5, 0.5], dtype=np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            reward = 1.0
            done = False
            
            step_result = StepResult(state, action, reward, next_state, done)
            policy.update(step_result)
        
        self.assertEqual(policy.steps, 10)


class TestAgent(unittest.TestCase):
    def test_agent_creation(self):
        env = LunarLanderCEnv(continuous=True)
        policy = A2CPolicy(state_dim=8, action_dim=2)
        agent = Agent(env, policy)
        self.assertEqual(agent.env, env)
        self.assertEqual(agent.policy, policy)
        env.close()

    def test_run_episode(self):
        env = LunarLanderCEnv(continuous=True, render_mode=None)
        policy = A2CPolicy(state_dim=8, action_dim=2)
        agent = Agent(env, policy)
        
        total_reward, steps = agent.run_episode(max_steps=10)
        self.assertIsInstance(total_reward, float)
        self.assertIsInstance(steps, int)
        self.assertGreater(steps, 0)
        self.assertLessEqual(steps, 10)
        env.close()


if __name__ == "__main__":
    unittest.main()
