"""
Unit tests for HalfCheetah RL logic
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from half_cheetah_logic import (
    HalfCheetahEnvironment,
    PPO,
    TD3,
    SAC,
    Agent
)


class TestHalfCheetahEnvironment(unittest.TestCase):
    """Test HalfCheetahEnvironment class"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = HalfCheetahEnvironment()
    
    def tearDown(self):
        """Clean up"""
        self.env.close()
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertIsNotNone(self.env.env)
        self.assertEqual(self.env.forward_reward_weight, 1.0)
        self.assertEqual(self.env.ctrl_cost_weight, 0.1)
    
    def test_reset(self):
        """Test environment reset"""
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape[0], self.env.get_state_dim())
    
    def test_step(self):
        """Test environment step"""
        self.env.reset()
        action = np.zeros(self.env.get_action_dim())
        next_state, reward, done, info = self.env.step(action)
        
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_dimensions(self):
        """Test state and action dimensions"""
        state_dim = self.env.get_state_dim()
        action_dim = self.env.get_action_dim()
        
        # HalfCheetah-v5 has 17 or 18 state dims depending on exclude_current_positions_from_observation
        self.assertIn(state_dim, [17, 18])
        self.assertEqual(action_dim, 6)  # HalfCheetah has 6 action dimensions
    
    def test_action_bounds(self):
        """Test action bounds"""
        low, high = self.env.get_action_bounds()
        self.assertIsInstance(low, np.ndarray)
        self.assertIsInstance(high, np.ndarray)
        self.assertEqual(len(low), self.env.get_action_dim())
        self.assertEqual(len(high), self.env.get_action_dim())


class TestPPO(unittest.TestCase):
    """Test PPO algorithm"""
    
    def setUp(self):
        """Set up test environment and policy"""
        self.env = HalfCheetahEnvironment()
        self.state_dim = self.env.get_state_dim()
        self.action_dim = self.env.get_action_dim()
        self.policy = PPO(
            self.state_dim,
            self.action_dim,
            hidden_dims=[64, 64],
            lr=3e-4
        )
    
    def tearDown(self):
        """Clean up"""
        self.env.close()
    
    def test_initialization(self):
        """Test PPO initialization"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
        self.assertIsNotNone(self.policy.network)
    
    def test_select_action(self):
        """Test action selection"""
        state = self.env.reset()
        action = self.policy.select_action(state)
        
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape[0], self.action_dim)
        self.assertTrue(np.all(action >= -1) and np.all(action <= 1))
    
    def test_store_transition(self):
        """Test storing transitions"""
        state = self.env.reset()
        action = self.policy.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        
        initial_len = len(self.policy.rewards)
        self.policy.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.policy.rewards), initial_len + 1)
    
    def test_training_episode(self):
        """Test training for one episode"""
        agent = Agent(self.env, self.policy)
        
        reward = agent.run_episode(train=True)
        self.assertIsInstance(reward, (float, np.floating, int))
        self.assertEqual(len(agent.episode_rewards), 1)


class TestTD3(unittest.TestCase):
    """Test TD3 algorithm"""
    
    def setUp(self):
        """Set up test environment and policy"""
        self.env = HalfCheetahEnvironment()
        self.state_dim = self.env.get_state_dim()
        self.action_dim = self.env.get_action_dim()
        self.policy = TD3(
            self.state_dim,
            self.action_dim,
            hidden_dims=[64, 64],
            lr=3e-4,
            batch_size=32
        )
    
    def tearDown(self):
        """Clean up"""
        self.env.close()
    
    def test_initialization(self):
        """Test TD3 initialization"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
        self.assertIsNotNone(self.policy.actor)
        self.assertIsNotNone(self.policy.critic1)
        self.assertIsNotNone(self.policy.critic2)
    
    def test_select_action(self):
        """Test action selection"""
        state = self.env.reset()
        action = self.policy.select_action(state)
        
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape[0], self.action_dim)
    
    def test_store_transition(self):
        """Test storing transitions"""
        state = self.env.reset()
        action = self.policy.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        
        initial_len = len(self.policy.memory)
        self.policy.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.policy.memory), initial_len + 1)
    
    def test_training_episode(self):
        """Test training for multiple episodes"""
        agent = Agent(self.env, self.policy)
        
        # Run multiple episodes to fill memory
        for _ in range(3):
            reward = agent.run_episode(train=True)
            self.assertIsInstance(reward, (float, np.floating, int))


class TestSAC(unittest.TestCase):
    """Test SAC algorithm"""
    
    def setUp(self):
        """Set up test environment and policy"""
        self.env = HalfCheetahEnvironment()
        self.state_dim = self.env.get_state_dim()
        self.action_dim = self.env.get_action_dim()
        self.policy = SAC(
            self.state_dim,
            self.action_dim,
            hidden_dims=[64, 64],
            lr=3e-4,
            batch_size=32
        )
    
    def tearDown(self):
        """Clean up"""
        self.env.close()
    
    def test_initialization(self):
        """Test SAC initialization"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
        self.assertIsNotNone(self.policy.actor)
        self.assertIsNotNone(self.policy.critic1)
        self.assertIsNotNone(self.policy.critic2)
    
    def test_select_action(self):
        """Test action selection"""
        state = self.env.reset()
        action = self.policy.select_action(state)
        
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape[0], self.action_dim)
    
    def test_store_transition(self):
        """Test storing transitions"""
        state = self.env.reset()
        action = self.policy.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        
        initial_len = len(self.policy.memory)
        self.policy.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.policy.memory), initial_len + 1)
    
    def test_training_episode(self):
        """Test training for multiple episodes"""
        agent = Agent(self.env, self.policy)
        
        # Run multiple episodes
        for i in range(5):
            reward = agent.run_episode(train=True)
            self.assertIsInstance(reward, (float, np.floating, int))
            print(f"Episode {i+1}: Reward = {reward:.2f}, Memory size = {len(self.policy.memory)}")


class TestAgent(unittest.TestCase):
    """Test Agent class"""
    
    def setUp(self):
        """Set up test environment and agent"""
        self.env = HalfCheetahEnvironment()
        self.policy = PPO(
            self.env.get_state_dim(),
            self.env.get_action_dim(),
            hidden_dims=[64, 64]
        )
        self.agent = Agent(self.env, self.policy)
    
    def tearDown(self):
        """Clean up"""
        self.env.close()
    
    def test_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.environment, self.env)
        self.assertEqual(self.agent.policy, self.policy)
        self.assertEqual(len(self.agent.episode_rewards), 0)
    
    def test_set_policy(self):
        """Test setting policy"""
        new_policy = SAC(
            self.env.get_state_dim(),
            self.env.get_action_dim(),
            hidden_dims=[64, 64],
            batch_size=32
        )
        self.agent.set_policy(new_policy)
        self.assertEqual(self.agent.policy, new_policy)
    
    def test_run_episode(self):
        """Test running episode"""
        reward = self.agent.run_episode(train=True)
        
        self.assertIsInstance(reward, (float, np.floating, int))
        self.assertEqual(len(self.agent.episode_rewards), 1)
        self.assertEqual(self.agent.current_episode_reward, reward)
    
    def test_get_returns(self):
        """Test getting returns"""
        for _ in range(3):
            self.agent.run_episode(train=True)
        
        returns = self.agent.get_returns()
        self.assertEqual(len(returns), 3)
        self.assertIsInstance(returns, list)
    
    def test_reset_stats(self):
        """Test resetting statistics"""
        self.agent.run_episode(train=True)
        self.agent.reset_stats()
        
        self.assertEqual(len(self.agent.episode_rewards), 0)
        self.assertEqual(self.agent.current_episode_reward, 0)


if __name__ == '__main__':
    unittest.main()
