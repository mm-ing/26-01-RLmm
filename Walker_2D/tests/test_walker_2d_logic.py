"""Unit tests for Walker2D RL logic"""
import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from walker_2d_logic import (
    Walker2DEnvironment, PPO, TD3, SAC, Agent, MLP, ActorCritic, ReplayBuffer
)


class TestWalker2DEnvironment(unittest.TestCase):
    """Test Walker2D environment wrapper"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = Walker2DEnvironment()
    
    def tearDown(self):
        """Clean up"""
        self.env.close()
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertIsNotNone(self.env.env)
        self.assertEqual(self.env.get_state_dim(), 17)
        self.assertEqual(self.env.get_action_dim(), 6)
    
    def test_reset(self):
        """Test environment reset"""
        state = self.env.reset()
        self.assertEqual(state.shape, (17,))
        self.assertTrue(np.all(np.isfinite(state)))
    
    def test_step(self):
        """Test environment step"""
        self.env.reset()
        action = np.zeros(6)
        state, reward, done, truncated, info = self.env.step(action)
        
        self.assertEqual(state.shape, (17,))
        self.assertIsInstance(reward, (int, float, np.number))
        self.assertIsInstance(done, (bool, np.bool_))
    
    def test_action_bounds(self):
        """Test action space bounds"""
        low, high = self.env.get_action_bounds()
        self.assertEqual(low.shape, (6,))
        self.assertEqual(high.shape, (6,))
        self.assertTrue(np.all(low == -1))
        self.assertTrue(np.all(high == 1))
    
    def test_custom_params(self):
        """Test custom environment parameters"""
        env = Walker2DEnvironment(
            forward_reward_weight=2.0,
            ctrl_cost_weight=0.002,
            healthy_reward=2.0
        )
        state = env.reset()
        self.assertEqual(state.shape, (17,))
        env.close()


class TestMLP(unittest.TestCase):
    """Test MLP network"""
    
    def test_initialization(self):
        """Test MLP initialization"""
        mlp = MLP(10, 5, [64, 64], activation='relu')
        self.assertIsNotNone(mlp)
    
    def test_forward(self):
        """Test forward pass"""
        import torch
        mlp = MLP(10, 5, [64, 64], activation='relu')
        x = torch.randn(32, 10)
        y = mlp(x)
        self.assertEqual(y.shape, (32, 5))
    
    def test_activations(self):
        """Test different activation functions"""
        import torch
        activations = ['relu', 'tanh', 'leaky_relu']
        for act in activations:
            mlp = MLP(10, 5, [32], activation=act)
            x = torch.randn(16, 10)
            y = mlp(x)
            self.assertEqual(y.shape, (16, 5))


class TestActorCritic(unittest.TestCase):
    """Test ActorCritic network"""
    
    def test_initialization(self):
        """Test ActorCritic initialization"""
        ac = ActorCritic(17, 6, [256, 256])
        self.assertIsNotNone(ac.actor)
        self.assertIsNotNone(ac.critic)
    
    def test_forward(self):
        """Test forward pass"""
        import torch
        ac = ActorCritic(17, 6, [256, 256])
        state = torch.randn(1, 17)
        mean, std, value = ac(state)
        
        self.assertEqual(mean.shape, (1, 6))
        self.assertEqual(std.shape, (6,))
        self.assertEqual(value.shape, (1, 1))
    
    def test_get_action(self):
        """Test action sampling"""
        import torch
        ac = ActorCritic(17, 6, [256, 256])
        state = torch.randn(1, 17)
        action, log_prob, value = ac.get_action(state)
        
        self.assertEqual(action.shape, (1, 6))
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(value.shape, (1, 1))


class TestPPO(unittest.TestCase):
    """Test PPO algorithm"""
    
    def setUp(self):
        """Set up test PPO"""
        self.ppo = PPO(17, 6, [64, 64])
    
    def test_initialization(self):
        """Test PPO initialization"""
        self.assertEqual(self.ppo.state_dim, 17)
        self.assertEqual(self.ppo.action_dim, 6)
        self.assertIsNotNone(self.ppo.policy)
    
    def test_select_action(self):
        """Test action selection"""
        state = np.random.randn(17)
        action, log_prob = self.ppo.select_action(state)
        
        self.assertEqual(action.shape, (6,))
        self.assertIsInstance(log_prob, (float, np.floating))
    
    def test_store_transition(self):
        """Test storing transitions"""
        state = np.random.randn(17)
        action = np.random.randn(6)
        reward = 1.0
        log_prob = 0.5
        value = 0.8
        done = False
        
        self.ppo.store_transition(state, action, reward, log_prob, value, done)
        self.assertEqual(len(self.ppo.states), 1)
    
    def test_train_step(self):
        """Test training step"""
        # Collect some transitions
        for _ in range(10):
            state = np.random.randn(17)
            action, log_prob = self.ppo.select_action(state)
            self.ppo.store_transition(state, action, 1.0, log_prob, 0.5, False)
        
        loss = self.ppo.train_step()
        self.assertIsInstance(loss, (float, np.floating))
        self.assertEqual(len(self.ppo.states), 0)  # Should be cleared


class TestReplayBuffer(unittest.TestCase):
    """Test ReplayBuffer"""
    
    def test_initialization(self):
        """Test buffer initialization"""
        buffer = ReplayBuffer(1000)
        self.assertEqual(len(buffer), 0)
    
    def test_push(self):
        """Test adding experiences"""
        buffer = ReplayBuffer(1000)
        buffer.push(np.zeros(17), np.zeros(6), 1.0, np.zeros(17), False)
        self.assertEqual(len(buffer), 1)
    
    def test_sample(self):
        """Test sampling"""
        buffer = ReplayBuffer(1000)
        for _ in range(100):
            buffer.push(np.random.randn(17), np.random.randn(6),
                       np.random.rand(), np.random.randn(17),
                       np.random.rand() > 0.9)
        
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones = batch
        
        self.assertEqual(states.shape, (32, 17))
        self.assertEqual(actions.shape, (32, 6))
        self.assertEqual(rewards.shape, (32,))
    
    def test_capacity(self):
        """Test buffer capacity"""
        buffer = ReplayBuffer(10)
        for i in range(20):
            buffer.push(np.zeros(17), np.zeros(6), float(i), np.zeros(17), False)
        
        self.assertEqual(len(buffer), 10)


class TestTD3(unittest.TestCase):
    """Test TD3 algorithm"""
    
    def setUp(self):
        """Set up test TD3"""
        self.td3 = TD3(17, 6, [64, 64])
    
    def test_initialization(self):
        """Test TD3 initialization"""
        self.assertEqual(self.td3.state_dim, 17)
        self.assertEqual(self.td3.action_dim, 6)
        self.assertIsNotNone(self.td3.actor)
        self.assertIsNotNone(self.td3.critic1)
        self.assertIsNotNone(self.td3.critic2)
    
    def test_select_action(self):
        """Test action selection"""
        state = np.random.randn(17)
        action = self.td3.select_action(state, noise=0.1)
        
        self.assertEqual(action.shape, (6,))
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
    
    def test_noise_types(self):
        """Test different noise types"""
        for noise_type in ['exploration', 'target_smoothing']:
            td3 = TD3(17, 6, [64, 64], noise_type=noise_type)
            state = np.random.randn(17)
            action = td3.select_action(state, noise=0.1)
            self.assertEqual(action.shape, (6,))
    
    def test_store_transition(self):
        """Test storing transitions"""
        state = np.random.randn(17)
        action = np.random.randn(6)
        reward = 1.0
        next_state = np.random.randn(17)
        done = False
        
        self.td3.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.td3.memory), 1)
    
    def test_train_step(self):
        """Test training step"""
        # Fill buffer
        for _ in range(300):
            self.td3.store_transition(
                np.random.randn(17), np.random.randn(6),
                np.random.rand(), np.random.randn(17),
                np.random.rand() > 0.9
            )
        
        loss = self.td3.train_step()
        self.assertIsInstance(loss, (float, np.floating))


class TestSAC(unittest.TestCase):
    """Test SAC algorithm"""
    
    def setUp(self):
        """Set up test SAC"""
        self.sac = SAC(17, 6, [64, 64])
    
    def test_initialization(self):
        """Test SAC initialization"""
        self.assertEqual(self.sac.state_dim, 17)
        self.assertEqual(self.sac.action_dim, 6)
        self.assertIsNotNone(self.sac.actor)
        self.assertIsNotNone(self.sac.critic1)
        self.assertIsNotNone(self.sac.critic2)
    
    def test_select_action(self):
        """Test action selection"""
        state = np.random.randn(17)
        action = self.sac.select_action(state, deterministic=False)
        
        self.assertEqual(action.shape, (6,))
        self.assertTrue(np.all(action >= -1))
        self.assertTrue(np.all(action <= 1))
    
    def test_deterministic_action(self):
        """Test deterministic action selection"""
        state = np.random.randn(17)
        action1 = self.sac.select_action(state, deterministic=True)
        action2 = self.sac.select_action(state, deterministic=True)
        
        np.testing.assert_array_almost_equal(action1, action2, decimal=5)
    
    def test_store_transition(self):
        """Test storing transitions"""
        state = np.random.randn(17)
        action = np.random.randn(6)
        reward = 1.0
        next_state = np.random.randn(17)
        done = False
        
        self.sac.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.sac.memory), 1)
    
    def test_train_step(self):
        """Test training step"""
        # Fill buffer
        for _ in range(300):
            self.sac.store_transition(
                np.random.randn(17), np.random.randn(6),
                np.random.rand(), np.random.randn(17),
                np.random.rand() > 0.9
            )
        
        loss = self.sac.train_step()
        self.assertIsInstance(loss, (float, np.floating))
    
    def test_auto_entropy_tuning(self):
        """Test automatic entropy tuning"""
        sac = SAC(17, 6, [64, 64], auto_entropy_tuning=True)
        self.assertTrue(sac.auto_entropy_tuning)
        self.assertIsNotNone(sac.log_alpha)


class TestAgent(unittest.TestCase):
    """Test Agent class"""
    
    def setUp(self):
        """Set up test agent"""
        self.env = Walker2DEnvironment()
        self.policy = PPO(17, 6, [64, 64])
        self.agent = Agent(self.env, self.policy)
    
    def tearDown(self):
        """Clean up"""
        self.env.close()
    
    def test_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent.env)
        self.assertIsNotNone(self.agent.policy)
        self.assertEqual(len(self.agent.rewards_history), 0)
    
    def test_run_episode_ppo(self):
        """Test running episode with PPO"""
        reward = self.agent.run_episode(train=True)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertTrue(np.isfinite(reward))
    
    def test_run_episode_td3(self):
        """Test running episode with TD3"""
        self.agent.policy = TD3(17, 6, [64, 64])
        reward = self.agent.run_episode(train=True)
        self.assertIsInstance(reward, (float, np.floating))
    
    def test_run_episode_sac(self):
        """Test running episode with SAC"""
        self.agent.policy = SAC(17, 6, [64, 64])
        reward = self.agent.run_episode(train=True)
        self.assertIsInstance(reward, (float, np.floating))
    
    def test_multiple_episodes(self):
        """Test running multiple episodes"""
        rewards = []
        for _ in range(3):
            reward = self.agent.run_episode(train=True)
            rewards.append(reward)
        
        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
    
    def test_train_method(self):
        """Test train method"""
        episode_data = []
        
        def callback(episode, reward):
            episode_data.append((episode, reward))
        
        rewards = self.agent.train(5, callback=callback)
        
        self.assertEqual(len(rewards), 5)
        self.assertEqual(len(episode_data), 5)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_ppo_training(self):
        """Test PPO training integration"""
        env = Walker2DEnvironment()
        policy = PPO(17, 6, [64, 64], epochs=5)
        agent = Agent(env, policy)
        
        rewards = []
        for _ in range(10):
            reward = agent.run_episode(train=True)
            rewards.append(reward)
        
        self.assertEqual(len(rewards), 10)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        env.close()
    
    def test_td3_training(self):
        """Test TD3 training integration"""
        env = Walker2DEnvironment()
        policy = TD3(17, 6, [64, 64], batch_size=64)
        agent = Agent(env, policy)
        
        rewards = []
        for _ in range(10):
            reward = agent.run_episode(train=True)
            rewards.append(reward)
        
        self.assertEqual(len(rewards), 10)
        env.close()
    
    def test_sac_training(self):
        """Test SAC training integration"""
        env = Walker2DEnvironment()
        policy = SAC(17, 6, [64, 64], batch_size=64)
        agent = Agent(env, policy)
        
        rewards = []
        for _ in range(10):
            reward = agent.run_episode(train=True)
            rewards.append(reward)
        
        self.assertEqual(len(rewards), 10)
        env.close()


if __name__ == '__main__':
    unittest.main()
