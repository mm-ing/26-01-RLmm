import unittest
import numpy as np
from lunar_lander_logic import (
    LunarLanderEnv,
    DQNPolicy,
    DDQNPolicy,
    PrioritizedDDQNPolicy,
    DuelingDDQNPolicy,
    NoisyDQNPolicy,
    DistributionalDQNPolicy,
    RainbowPolicy,
    Agent,
)


class TestLunarLanderEnv(unittest.TestCase):
    def test_discrete_env_creation(self):
        """Test creation of discrete LunarLander environment."""
        env = LunarLanderEnv(continuous=False)
        self.assertIsNotNone(env)
        self.assertEqual(env.total_actions, 4)
        env.close()

    def test_continuous_env_creation(self):
        """Test creation of continuous LunarLander environment."""
        env = LunarLanderEnv(continuous=True)
        self.assertIsNotNone(env)
        self.assertEqual(env.total_actions, 25)  # 5^2 discretized actions
        env.close()

    def test_env_reset(self):
        """Test environment reset."""
        env = LunarLanderEnv()
        state = env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (8,))
        env.close()

    def test_env_step(self):
        """Test environment step."""
        env = LunarLanderEnv()
        env.reset()
        action = env.action_from_index(0)
        next_state, reward, done, info = env.step(action)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        env.close()

    def test_action_from_index_discrete(self):
        """Test action conversion for discrete environment."""
        env = LunarLanderEnv(continuous=False)
        for i in range(4):
            action = env.action_from_index(i)
            self.assertIsInstance(action, int)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 4)
        env.close()

    def test_action_from_index_continuous(self):
        """Test action conversion for continuous environment."""
        env = LunarLanderEnv(continuous=True)
        for i in range(25):
            action = env.action_from_index(i)
            self.assertIsInstance(action, np.ndarray)
            self.assertEqual(action.shape, (2,))
        env.close()


class TestPolicies(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 8
        self.action_dim = 4
        self.alpha = 0.001
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.001
        self.hidden_layers = 2
        self.hidden_units = 64
        self.batch_size = 32
        self.replay_size = 1000
        self.activation = "ReLU"

    def test_dqn_policy_creation(self):
        """Test DQN policy creation."""
        policy = DQNPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
        )
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "DQN")

    def test_ddqn_policy_creation(self):
        """Test DDQN policy creation."""
        policy = DDQNPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
        )
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "DDQN")

    def test_prioritized_ddqn_policy_creation(self):
        """Test Prioritized DDQN policy creation."""
        policy = PrioritizedDDQNPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
        )
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "Prioritized DDQN")

    def test_dueling_ddqn_policy_creation(self):
        """Test Dueling DDQN policy creation."""
        policy = DuelingDDQNPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
        )
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "Dueling DDQN")

    def test_noisy_dqn_policy_creation(self):
        """Test Noisy DQN policy creation."""
        policy = NoisyDQNPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
        )
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "Noisy DQN")

    def test_distributional_dqn_policy_creation(self):
        """Test Distributional DQN policy creation."""
        policy = DistributionalDQNPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
        )
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "Distributional DQN")

    def test_rainbow_policy_creation(self):
        """Test Rainbow policy creation."""
        policy = RainbowPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
        )
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "Rainbow")

    def test_policy_select_action(self):
        """Test policy action selection."""
        policy = DQNPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
        )
        state = np.random.randn(self.state_dim).astype(np.float32)
        action = policy.select_action(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)

    def test_policy_store_and_learn(self):
        """Test policy store and learn."""
        policy = DQNPolicy(
            self.state_dim,
            self.action_dim,
            self.alpha,
            self.gamma,
            self.eps_start,
            self.eps_end,
            self.eps_decay,
            self.hidden_layers,
            self.hidden_units,
            self.batch_size,
            self.replay_size,
            self.activation,
            warmup_steps=50,
        )
        
        # Store some experiences
        for _ in range(100):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randint(self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = False
            policy.store(state, action, reward, next_state, done)
        
        # Check if learning is possible
        self.assertTrue(policy.can_learn())
        
        # Try learning
        policy.learn()  # Should not raise exception


class TestAgent(unittest.TestCase):
    def test_agent_creation(self):
        """Test agent creation."""
        env = LunarLanderEnv()
        policy = DQNPolicy(8, 4, 0.001, 0.99, 1.0, 0.01, 0.001, 2, 64, 32, 1000, "ReLU")
        agent = Agent(env, policy)
        self.assertIsNotNone(agent)
        env.close()

    def test_agent_run_episode(self):
        """Test agent running an episode."""
        env = LunarLanderEnv()
        policy = DQNPolicy(8, 4, 0.001, 0.99, 1.0, 0.01, 0.001, 2, 64, 32, 1000, "ReLU", warmup_steps=10)
        agent = Agent(env, policy)
        
        total_reward = agent.run_episode(episode_idx=0, max_steps=100, step_delay=0.0)
        
        self.assertIsInstance(total_reward, float)
        env.close()


if __name__ == "__main__":
    unittest.main()
