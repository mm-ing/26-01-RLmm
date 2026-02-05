"""
BipedalWalker Environment and RL Algorithms
Implements Rainbow, A2C, TRPO, PPO, and SAC algorithms for BipedalWalker-v3
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from threading import Lock


class BipedalWalkerEnvironment:
    """Wrapper for BipedalWalker-v3 environment"""
    
    def __init__(self, hardcore=False, render_mode=None):
        self.hardcore = hardcore
        self.render_mode = render_mode
        self.env = None
        self.reset_env()
        
    def reset_env(self):
        """Reset or create environment"""
        if self.env:
            self.env.close()
        self.env = gym.make('BipedalWalker-v3', hardcore=self.hardcore, render_mode=self.render_mode)
        
    def reset(self):
        """Reset environment"""
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        """Take a step in the environment"""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info
    
    def get_state_dim(self):
        """Get state dimension"""
        return self.env.observation_space.shape[0]
    
    def get_action_dim(self):
        """Get action dimension"""
        return self.env.action_space.shape[0]
    
    def get_action_bounds(self):
        """Get action bounds"""
        return self.env.action_space.low, self.env.action_space.high
    
    def render(self):
        """Render environment and return RGB array"""
        if self.env and self.render_mode == 'rgb_array':
            return self.env.render()
        return None
    
    def close(self):
        """Close environment"""
        if self.env:
            self.env.close()


class MLP(nn.Module):
    """Multi-Layer Perceptron for policy/value networks"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], activation='relu'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


class ActorCritic(nn.Module):
    """Actor-Critic network for policy gradient methods"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(ActorCritic, self).__init__()
        
        # Actor (policy) network
        self.actor_layers = nn.ModuleList()
        dims = [state_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.actor_layers.append(nn.Linear(dims[i], dims[i+1]))
        
        self.mean_layer = nn.Linear(dims[-1], action_dim)
        self.log_std_layer = nn.Linear(dims[-1], action_dim)
        
        # Critic (value) network
        self.critic_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.critic_layers.append(nn.Linear(dims[i], dims[i+1]))
        self.value_layer = nn.Linear(dims[-1], 1)
    
    def forward(self, state):
        # Actor
        x = state
        for layer in self.actor_layers:
            x = F.relu(layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        # Critic
        x = state
        for layer in self.critic_layers:
            x = F.relu(layer(x))
        value = self.value_layer(x)
        
        return mean, log_std, value


class RainbowDQN:
    """Rainbow DQN (adapted for continuous actions using discretization)"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], lr=3e-4, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=64, n_actions=11):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_actions = n_actions  # Discretize each action dimension
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Discretized action space
        self.discrete_actions = self._create_discrete_actions()
        
        # Q-networks
        output_dim = len(self.discrete_actions)
        self.q_network = MLP(state_dim, output_dim, hidden_dims).to(self.device)
        self.target_network = MLP(state_dim, output_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.lock = Lock()
        
    def _create_discrete_actions(self):
        """Create discretized action space"""
        # Create a grid of discrete actions
        action_values = np.linspace(-1, 1, self.n_actions)
        actions = []
        for a1 in action_values:
            for a2 in action_values:
                for a3 in action_values:
                    for a4 in action_values:
                        actions.append([a1, a2, a3, a4])
        return np.array(actions)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(len(self.discrete_actions))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax(1).item()
        
        return self.discrete_actions[action_idx]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        # Find closest discrete action
        action_idx = np.argmin(np.sum((self.discrete_actions - action)**2, axis=1))
        with self.lock:
            self.memory.append((state, action_idx, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        with self.lock:
            batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())


class A2C:
    """Advantage Actor-Critic"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], lr=3e-4, 
                 gamma=0.99, value_coef=0.5, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.lock = Lock()
        
    def select_action(self, state):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std, value = self.network(state_tensor)
            std = log_std.exp()
            
            # Sample action from normal distribution
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        with self.lock:
            self.states.append(state)
            self.actions.append(action.cpu().numpy()[0])
            self.values.append(value.item())
            self.log_probs.append(log_prob.item())
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store reward"""
        with self.lock:
            self.rewards.append(reward)
    
    def train_step(self):
        """Perform one training step"""
        with self.lock:
            if len(self.rewards) == 0:
                return 0.0
            
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            rewards = self.rewards.copy()
            values = self.values.copy()
            old_log_probs = self.log_probs.copy()
            
            # Clear buffers
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
        
        # Compute returns and advantages
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        advantages = returns - values_tensor
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute new values and log probs
        mean, log_std, new_values = self.network(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Compute losses
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(new_values.squeeze(), returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()


class TRPO:
    """Trust Region Policy Optimization"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], lr=3e-4,
                 gamma=0.99, max_kl=0.01, damping=0.1, value_coef=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.max_kl = max_kl
        self.damping = damping
        self.value_coef = value_coef
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.value_optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.lock = Lock()
        
    def select_action(self, state):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std, _ = self.network(state_tensor)
            std = log_std.exp()
            
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
        
        with self.lock:
            self.states.append(state)
            self.actions.append(action.cpu().numpy()[0])
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store reward"""
        with self.lock:
            self.rewards.append(reward)
    
    def train_step(self):
        """Perform one training step (simplified TRPO)"""
        with self.lock:
            if len(self.rewards) == 0:
                return 0.0
            
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            rewards = self.rewards.copy()
            
            # Clear buffers
            self.states = []
            self.actions = []
            self.rewards = []
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Get old policy
        with torch.no_grad():
            old_mean, old_log_std, old_values = self.network(states)
            old_std = old_log_std.exp()
            old_dist = torch.distributions.Normal(old_mean, old_std)
            old_log_probs = old_dist.log_prob(actions).sum(dim=-1)
        
        # Compute advantages
        advantages = returns - old_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy with simplified TRPO (using clipping instead of conjugate gradient)
        mean, log_std, values = self.network(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Policy loss with KL constraint (simplified)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -(ratio * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss
        
        # Update
        self.value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.value_optimizer.step()
        
        return loss.item()


class PPO:
    """Proximal Policy Optimization"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], lr=3e-4,
                 gamma=0.99, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 update_epochs=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.lock = Lock()
        
    def select_action(self, state):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std, _ = self.network(state_tensor)
            std = log_std.exp()
            
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        with self.lock:
            self.states.append(state)
            self.actions.append(action.cpu().numpy()[0])
            self.log_probs.append(log_prob.item())
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store reward"""
        with self.lock:
            self.rewards.append(reward)
    
    def train_step(self):
        """Perform one training step"""
        with self.lock:
            if len(self.rewards) == 0:
                return 0.0
            
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            rewards = self.rewards.copy()
            old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
            
            # Clear buffers
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Multiple epochs of updates
        total_loss = 0
        for _ in range(self.update_epochs):
            mean, log_std, values = self.network(states)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Compute advantages
            advantages = returns - values.squeeze().detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.update_epochs


class SAC:
    """Soft Actor-Critic"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, buffer_size=100000, batch_size=64,
                 train_freq=4, gradient_steps=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.train_freq = train_freq  # Train every N steps
        self.gradient_steps = gradient_steps  # Number of gradient updates per training
        self.training_started = False  # Track if training has begun
        self.total_steps = 0  # Track total steps for training frequency
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network
        self.actor = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Critic networks (twin Q-functions)
        self.critic1 = MLP(state_dim + action_dim, 1, hidden_dims).to(self.device)
        self.critic2 = MLP(state_dim + action_dim, 1, hidden_dims).to(self.device)
        
        # Target critic networks
        self.target_critic1 = MLP(state_dim + action_dim, 1, hidden_dims).to(self.device)
        self.target_critic2 = MLP(state_dim + action_dim, 1, hidden_dims).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.memory = deque(maxlen=buffer_size)
        self.lock = Lock()
        
    def select_action(self, state, evaluate=False):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std, _ = self.actor(state_tensor)
            std = log_std.exp()
            
            if evaluate:
                action = torch.tanh(mean)
            else:
                dist = torch.distributions.Normal(mean, std)
                z = dist.sample()
                action = torch.tanh(z)
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Mark that training has started
        if not self.training_started:
            self.training_started = True
            print(f"SAC training started with {len(self.memory)} samples in memory")
        
        with self.lock:
            batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_mean, next_log_std, _ = self.actor(next_states)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            
            # Sample next actions with reparameterization
            next_z = next_dist.rsample()
            next_actions = torch.tanh(next_z)
            
            # Compute log probability with tanh correction
            next_log_probs = next_dist.log_prob(next_z).sum(dim=-1, keepdim=True)
            # Correct for tanh squashing
            next_log_probs -= torch.sum(torch.log(1 - next_actions.pow(2) + 1e-6), dim=-1, keepdim=True)
            
            next_state_actions = torch.cat([next_states, next_actions], dim=1)
            target_q1 = self.target_critic1(next_state_actions)
            target_q2 = self.target_critic2(next_state_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
        
        state_actions = torch.cat([states, actions], dim=1)
        current_q1 = self.critic1(state_actions)
        current_q2 = self.critic2(state_actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # Update actor
        # Freeze critic networks during actor update
        for param in self.critic1.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
        
        mean, log_std, _ = self.actor(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        
        # Sample actions with reparameterization trick
        z = dist.rsample()
        new_actions = torch.tanh(z)
        
        # Compute log probability with tanh correction
        log_probs = dist.log_prob(z).sum(dim=-1, keepdim=True)
        # Correct for tanh squashing
        log_probs -= torch.sum(torch.log(1 - new_actions.pow(2) + 1e-6), dim=-1, keepdim=True)
        
        new_state_actions = torch.cat([states, new_actions], dim=1)
        q1 = self.critic1(new_state_actions)
        q2 = self.critic2(new_state_actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Unfreeze critic networks
        for param in self.critic1.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True
        
        # Soft update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return (critic1_loss.item() + critic2_loss.item() + actor_loss.item()) / 3


class Agent:
    """Agent for learning with different RL algorithms"""
    
    def __init__(self, environment, policy):
        self.environment = environment
        self.policy = policy
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def set_policy(self, policy):
        """Set the current policy"""
        self.policy = policy
        
    def run_episode(self, render=False, train=True, render_callback=None):
        """Run one episode"""
        state = self.environment.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 1600:
            # Render if callback provided (skip some frames for performance)
            if render and render_callback and steps % 2 == 0:  # Render every 2 steps
                frame = self.environment.render()
                if frame is not None:
                    render_callback(frame)
            
            action = self.policy.select_action(state)
            next_state, reward, done, _ = self.environment.step(action)
            
            if train:
                self.policy.store_transition(state, action, reward, next_state, done)
                
                # Train step for some algorithms
                if hasattr(self.policy, 'train_step'):
                    if isinstance(self.policy, (RainbowDQN, SAC)):
                        # For SAC, train every N steps after warmup
                        if isinstance(self.policy, SAC):
                            self.policy.total_steps += 1
                            if (len(self.policy.memory) >= self.policy.batch_size * 2 and 
                                self.policy.total_steps % self.policy.train_freq == 0):
                                # Perform multiple gradient steps per training
                                for _ in range(self.policy.gradient_steps):
                                    self.policy.train_step()
                        else:
                            if steps % 4 == 0:  # Train every 4 steps for Rainbow
                                self.policy.train_step()
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Train step for on-policy algorithms
        if train and hasattr(self.policy, 'train_step'):
            if isinstance(self.policy, (A2C, TRPO, PPO)):
                self.policy.train_step()
        
        # Update target network for Rainbow
        if train and isinstance(self.policy, RainbowDQN):
            if len(self.episode_rewards) % 10 == 0:
                self.policy.update_target_network()
        
        self.episode_rewards.append(episode_reward)
        self.current_episode_reward = episode_reward
        
        return episode_reward
    
    def get_returns(self):
        """Get episode returns"""
        return self.episode_rewards.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.episode_rewards = []
        self.current_episode_reward = 0
