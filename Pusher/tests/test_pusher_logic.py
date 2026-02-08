"""
Unit tests for Pusher RL logic
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pusher_logic import PusherEnvironment, DDPG, TD3, SAC, Agent


class TestPusherEnvironment(unittest.TestCase):
    """Test PusherEnvironment class"""

    def setUp(self):
        self.env = PusherEnvironment()

    def tearDown(self):
        self.env.close()

    def test_initialization(self):
        self.assertIsNotNone(self.env.env)
        self.assertEqual(self.env.reward_near_weight, 0.5)
        self.assertEqual(self.env.reward_dist_weight, 1.0)

    def test_reset(self):
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertGreater(self.env.get_state_dim(), 0)

    def test_step(self):
        self.env.reset()
        action = np.zeros(self.env.get_action_dim())
        next_state, reward, done, info = self.env.step(action)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_action_bounds(self):
        low, high = self.env.get_action_bounds()
        self.assertEqual(len(low), self.env.get_action_dim())
        self.assertEqual(len(high), self.env.get_action_dim())

    def test_scale_action(self):
        action = np.zeros(self.env.get_action_dim())
        scaled = self.env.scale_action(action)
        low, high = self.env.get_action_bounds()
        self.assertTrue(np.all(scaled >= low) and np.all(scaled <= high))


class TestDDPG(unittest.TestCase):
    def setUp(self):
        self.env = PusherEnvironment()
        self.policy = DDPG(self.env.get_state_dim(), self.env.get_action_dim(), hidden_dims=[64, 64])

    def tearDown(self):
        self.env.close()

    def test_select_action(self):
        state = self.env.reset()
        action = self.policy.select_action(state)
        self.assertEqual(action.shape[0], self.env.get_action_dim())
        self.assertTrue(np.all(action >= -1) and np.all(action <= 1))

    def test_store_transition(self):
        state = self.env.reset()
        action = self.policy.select_action(state)
        next_state, reward, done, _ = self.env.step(self.env.scale_action(action))
        initial_len = len(self.policy.memory)
        self.policy.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.policy.memory), initial_len + 1)

    def test_training_episode(self):
        agent = Agent(self.env, self.policy)
        reward, _ = agent.run_episode(train=True)
        self.assertIsInstance(reward, (float, np.floating, int))


class TestTD3(unittest.TestCase):
    def setUp(self):
        self.env = PusherEnvironment()
        self.policy = TD3(self.env.get_state_dim(), self.env.get_action_dim(), hidden_dims=[64, 64], batch_size=32)

    def tearDown(self):
        self.env.close()

    def test_select_action(self):
        state = self.env.reset()
        action = self.policy.select_action(state)
        self.assertEqual(action.shape[0], self.env.get_action_dim())

    def test_store_transition(self):
        state = self.env.reset()
        action = self.policy.select_action(state)
        next_state, reward, done, _ = self.env.step(self.env.scale_action(action))
        initial_len = len(self.policy.memory)
        self.policy.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.policy.memory), initial_len + 1)

    def test_training_episode(self):
        agent = Agent(self.env, self.policy)
        for _ in range(2):
            reward, _ = agent.run_episode(train=True)
            self.assertIsInstance(reward, (float, np.floating, int))


class TestSAC(unittest.TestCase):
    def setUp(self):
        self.env = PusherEnvironment()
        self.policy = SAC(self.env.get_state_dim(), self.env.get_action_dim(), hidden_dims=[64, 64], batch_size=32)

    def tearDown(self):
        self.env.close()

    def test_select_action(self):
        state = self.env.reset()
        action = self.policy.select_action(state)
        self.assertEqual(action.shape[0], self.env.get_action_dim())

    def test_store_transition(self):
        state = self.env.reset()
        action = self.policy.select_action(state)
        next_state, reward, done, _ = self.env.step(self.env.scale_action(action))
        initial_len = len(self.policy.memory)
        self.policy.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.policy.memory), initial_len + 1)

    def test_training_episode(self):
        agent = Agent(self.env, self.policy)
        for _ in range(2):
            reward, _ = agent.run_episode(train=True)
            self.assertIsInstance(reward, (float, np.floating, int))


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.env = PusherEnvironment()
        self.policy = DDPG(self.env.get_state_dim(), self.env.get_action_dim(), hidden_dims=[64, 64])
        self.agent = Agent(self.env, self.policy)

    def tearDown(self):
        self.env.close()

    def test_set_policy(self):
        new_policy = SAC(self.env.get_state_dim(), self.env.get_action_dim(), hidden_dims=[64, 64])
        self.agent.set_policy(new_policy)
        self.assertEqual(self.agent.policy, new_policy)

    def test_run_episode(self):
        reward, _ = self.agent.run_episode(train=True)
        self.assertIsInstance(reward, (float, np.floating, int))
        self.assertEqual(len(self.agent.episode_rewards), 1)

    def test_get_returns(self):
        for _ in range(2):
            self.agent.run_episode(train=True)
        returns = self.agent.get_returns()
        self.assertEqual(len(returns), 2)

    def test_reset_stats(self):
        self.agent.run_episode(train=True)
        self.agent.reset_stats()
        self.assertEqual(len(self.agent.episode_rewards), 0)
        self.assertEqual(self.agent.current_episode_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
