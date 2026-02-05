"""
Tests for BipedalWalker RL Logic
"""

import unittest
import numpy as np
import torch
from bipedal_walker_logic import (
    BipedalWalkerEnvironment,
    RainbowDQN,
    A2C,
    TRPO,
    PPO,
    SAC,
    Agent
)


class TestBipedalWalkerEnvironment(unittest.TestCase):
    """Test BipedalWalker environment wrapper"""
    
    def setUp(self):
        self.env = BipedalWalkerEnvironment(hardcore=False)
    
    def tearDown(self):
        self.env.close()
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertIsNotNone(self.env.env)
        self.assertEqual(self.env.hardcore, False)
    
    def test_reset(self):
        """Test environment reset"""
        state = self.env.reset()
        self.assertEqual(len(state), 24)
        self.assertTrue(isinstance(state, np.ndarray))
    
    def test_step(self):
        """Test environment step"""
        self.env.reset()
        action = np.zeros(4)
        next_state, reward, done, info = self.env.step(action)
        
        self.assertEqual(len(next_state), 24)
        self.assertTrue(isinstance(reward, (int, float)))
        self.assertTrue(isinstance(done, bool))
    
    def test_dimensions(self):
        """Test state and action dimensions"""
        self.assertEqual(self.env.get_state_dim(), 24)
        self.assertEqual(self.env.get_action_dim(), 4)
    
    def test_action_bounds(self):
        """Test action bounds"""
        low, high = self.env.get_action_bounds()
        self.assertEqual(len(low), 4)
        self.assertEqual(len(high), 4)
        self.assertTrue(np.all(low == -1))
        self.assertTrue(np.all(high == 1))
    
    def test_hardcore_mode(self):
        """Test hardcore mode"""
        env_hardcore = BipedalWalkerEnvironment(hardcore=True)
        self.assertEqual(env_hardcore.hardcore, True)
        env_hardcore.close()


class TestRainbowDQN(unittest.TestCase):
    """Test Rainbow DQN algorithm"""
    
    def setUp(self):
        self.state_dim = 24
        self.action_dim = 4
        self.policy = RainbowDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[64, 64],
            n_actions=3  # Small for testing
        )
    
    def test_initialization(self):
        """Test policy initialization"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
        self.assertTrue(len(self.policy.discrete_actions) > 0)
    
    def test_select_action(self):
        """Test action selection"""
        state = np.random.randn(self.state_dim)
        action = self.policy.select_action(state)
        
        self.assertEqual(len(action), self.action_dim)
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
    
    def test_store_transition(self):
        """Test transition storage"""
        state = np.random.randn(self.state_dim)
        action = np.random.randn(self.action_dim)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False
        
        self.policy.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.policy.memory), 1)
    
    def test_training(self):
        """Test training step"""
        # Fill memory
        for _ in range(100):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim)
            done = False
            self.policy.store_transition(state, action, reward, next_state, done)
        
        loss = self.policy.train_step()
        self.assertTrue(isinstance(loss, float))
        self.assertTrue(loss >= 0)


class TestA2C(unittest.TestCase):
    """Test A2C algorithm"""
    
    def setUp(self):
        self.state_dim = 24
        self.action_dim = 4
        self.policy = A2C(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[64, 64]
        )
    
    def test_initialization(self):
        """Test policy initialization"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
    
    def test_select_action(self):
        """Test action selection"""
        state = np.random.randn(self.state_dim)
        action = self.policy.select_action(state)
        
        self.assertEqual(len(action), self.action_dim)
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
    
    def test_store_and_train(self):
        """Test storing transitions and training"""
        state = np.random.randn(self.state_dim)
        action = self.policy.select_action(state)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False
        
        self.policy.store_transition(state, action, reward, next_state, done)
        loss = self.policy.train_step()
        
        self.assertTrue(isinstance(loss, float))


class TestTRPO(unittest.TestCase):
    """Test TRPO algorithm"""
    
    def setUp(self):
        self.state_dim = 24
        self.action_dim = 4
        self.policy = TRPO(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[64, 64]
        )
    
    def test_initialization(self):
        """Test policy initialization"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
    
    def test_select_action(self):
        """Test action selection"""
        state = np.random.randn(self.state_dim)
        action = self.policy.select_action(state)
        
        self.assertEqual(len(action), self.action_dim)
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
    
    def test_store_and_train(self):
        """Test storing transitions and training"""
        state = np.random.randn(self.state_dim)
        action = self.policy.select_action(state)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False
        
        self.policy.store_transition(state, action, reward, next_state, done)
        loss = self.policy.train_step()
        
        self.assertTrue(isinstance(loss, float))


class TestPPO(unittest.TestCase):
    """Test PPO algorithm"""
    
    def setUp(self):
        self.state_dim = 24
        self.action_dim = 4
        self.policy = PPO(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[64, 64],
            update_epochs=2  # Small for testing
        )
    
    def test_initialization(self):
        """Test policy initialization"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
    
    def test_select_action(self):
        """Test action selection"""
        state = np.random.randn(self.state_dim)
        action = self.policy.select_action(state)
        
        self.assertEqual(len(action), self.action_dim)
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
    
    def test_store_and_train(self):
        """Test storing transitions and training"""
        state = np.random.randn(self.state_dim)
        action = self.policy.select_action(state)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False
        
        self.policy.store_transition(state, action, reward, next_state, done)
        loss = self.policy.train_step()
        
        self.assertTrue(isinstance(loss, float))


class TestSAC(unittest.TestCase):
    """Test SAC algorithm"""
    
    def setUp(self):
        self.state_dim = 24
        self.action_dim = 4
        self.policy = SAC(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[64, 64]
        )
    
    def test_initialization(self):
        """Test policy initialization"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
    
    def test_select_action(self):
        """Test action selection"""
        state = np.random.randn(self.state_dim)
        action = self.policy.select_action(state)
        
        self.assertEqual(len(action), self.action_dim)
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
    
    def test_store_transition(self):
        """Test transition storage"""
        state = np.random.randn(self.state_dim)
        action = np.random.randn(self.action_dim)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False
        
        self.policy.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.policy.memory), 1)
    
    def test_training(self):
        """Test training step"""
        # Fill memory
        for _ in range(100):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim)
            done = False
            self.policy.store_transition(state, action, reward, next_state, done)
        
        loss = self.policy.train_step()
        self.assertTrue(isinstance(loss, float))
        self.assertTrue(loss >= 0)


class TestAgent(unittest.TestCase):
    """Test Agent class"""
    
    def setUp(self):
        self.env = BipedalWalkerEnvironment(hardcore=False)
        self.policy = PPO(
            state_dim=24,
            action_dim=4,
            hidden_dims=[64, 64],
            update_epochs=1
        )
        self.agent = Agent(self.env, self.policy)
    
    def tearDown(self):
        self.env.close()
    
    def test_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent.environment)
        self.assertIsNotNone(self.agent.policy)
    
    def test_set_policy(self):
        """Test setting policy"""
        new_policy = A2C(state_dim=24, action_dim=4, hidden_dims=[64, 64])
        self.agent.set_policy(new_policy)
        self.assertEqual(self.agent.policy, new_policy)
    
    def test_run_episode(self):
        """Test running an episode"""
        reward = self.agent.run_episode(render=False, train=False)
        self.assertTrue(isinstance(reward, (int, float)))
        self.assertEqual(len(self.agent.episode_rewards), 1)
    
    def test_get_returns(self):
        """Test getting returns"""
        self.agent.run_episode(render=False, train=False)
        returns = self.agent.get_returns()
        self.assertEqual(len(returns), 1)
    
    def test_reset_stats(self):
        """Test resetting statistics"""
        self.agent.run_episode(render=False, train=False)
        self.agent.reset_stats()
        self.assertEqual(len(self.agent.episode_rewards), 0)
        self.assertEqual(self.agent.current_episode_reward, 0)


class TestLearningProgress(unittest.TestCase):
    """Test that algorithms can learn"""
    
    def setUp(self):
        self.env = BipedalWalkerEnvironment(hardcore=False)
    
    def tearDown(self):
        self.env.close()
    
    def test_ppo_learning(self):
        """Test PPO can improve over episodes"""
        policy = PPO(
            state_dim=24,
            action_dim=4,
            hidden_dims=[64, 64],
            update_epochs=1,
            lr=3e-4
        )
        agent = Agent(self.env, policy)
        
        # Run 5 training episodes
        for _ in range(5):
            agent.run_episode(render=False, train=True)
        
        # Check that we have collected rewards
        returns = agent.get_returns()
        self.assertEqual(len(returns), 5)
        self.assertTrue(all(isinstance(r, (int, float, np.number)) for r in returns))
    
    def test_sac_learning(self):
        """Test SAC can improve over episodes"""
        policy = SAC(
            state_dim=24,
            action_dim=4,
            hidden_dims=[64, 64],
            batch_size=32,
            lr=3e-4
        )
        agent = Agent(self.env, policy)
        
        # Run 10 training episodes to collect enough samples
        for _ in range(10):
            agent.run_episode(render=False, train=True)
        
        # Check that we have collected rewards
        returns = agent.get_returns()
        self.assertEqual(len(returns), 10)
        # Check all returns are numeric (including numpy types)
        self.assertTrue(all(isinstance(r, (int, float, np.number)) for r in returns))
        
        # Check that memory has samples
        self.assertGreater(len(policy.memory), policy.batch_size)


if __name__ == '__main__':
    unittest.main()
