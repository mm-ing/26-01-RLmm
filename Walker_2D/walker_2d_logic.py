"""Walker2D RL Logic - Environment wrapper and RL algorithms (PPO, TD3, SAC)"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Walker2DEnvironment:
    """Wrapper for Walker2d-v5 environment with configurable parameters"""
    
    def __init__(self, render_mode=None, forward_reward_weight=1.0, ctrl_cost_weight=0.001,
                 healthy_reward=1.0, terminate_when_unhealthy=True, healthy_z_range=(0.8, 2.0),
                 healthy_angle_range=(-1.0, 1.0), reset_noise_scale=0.005):
        """
        Initialize Walker2D environment
        
        Args:
            render_mode: 'rgb_array' for rendering, None for no rendering
            forward_reward_weight: Weight for forward movement reward
            ctrl_cost_weight: Weight for control cost penalty
            healthy_reward: Reward for staying healthy
            terminate_when_unhealthy: Whether to terminate when unhealthy
            healthy_z_range: Acceptable height range (min, max)
            healthy_angle_range: Acceptable angle range (min, max)
            reset_noise_scale: Scale of noise added on reset
        """
        self.env = gym.make(
            'Walker2d-v5',
            render_mode=render_mode,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            healthy_angle_range=healthy_angle_range,
            reset_noise_scale=reset_noise_scale
        )
        self.render_mode = render_mode
        
    def reset(self):
        """Reset environment and return initial state"""
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        """
        Take a step in the environment
        
        Returns:
            state, reward, done, truncated, info
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return state, reward, done, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return None
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def get_state_dim(self):
        """Get state dimension"""
        return self.env.observation_space.shape[0]
    
    def get_action_dim(self):
        """Get action dimension"""
        return self.env.action_space.shape[0]
    
    def get_action_bounds(self):
        """Get action space bounds"""
        return self.env.action_space.low, self.env.action_space.high


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, input_dim, output_dim, hidden_sizes, activation='relu'):
        super(MLP, self).__init__()
        self.activation_name = activation
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with orthogonal initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes, activation='tanh'):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = MLP(state_dim, action_dim, hidden_sizes, activation)
        
        # Critic network
        self.critic = MLP(state_dim, 1, hidden_sizes, activation)
        
        # Log std for continuous actions
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """Get action distribution and value"""
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        value = self.critic(state)
        return mean, std, value
    
    def get_action(self, state):
        """Sample action from policy"""
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value


class PPO:
    """Proximal Policy Optimization"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256],
                 lr=3e-4, gamma=0.99, clip_ratio=0.2, epochs=10,
                 batch_size=64, activation='tanh'):
        """
        Initialize PPO
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: Hidden layer sizes
            lr: Learning rate
            gamma: Discount factor
            clip_ratio: PPO clipping parameter
            epochs: Number of optimization epochs
            batch_size: Batch size for training
            activation: Activation function
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Actor-Critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_sizes, activation)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Tracking
        self.episode_count = 0
        self.nan_count = 0
        
    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.policy.get_action(state_tensor)
            
            # Check for NaN
            if torch.isnan(action).any() or torch.isnan(log_prob).any():
                logger.warning("NaN detected in PPO action selection")
                self.nan_count += 1
                # Return zero action as fallback
                return np.zeros(self.action_dim), 0.0
            
            return action.squeeze(0).numpy(), log_prob.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def train_step(self):
        """Perform PPO update"""
        if len(self.states) == 0:
            return 0.0
        
        self.episode_count += 1
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        
        # Compute returns and advantages
        returns = self._compute_returns()
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get values
        _, _, values = self.policy(states)
        values = values.squeeze()
        
        # Compute advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0.0
        dataset_size = len(self.states)
        
        for _ in range(self.epochs):
            # Mini-batch updates
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                mean, std, value = self.policy(batch_states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # Ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(value.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Check for NaN
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected in PPO at episode {self.episode_count}")
                    self.nan_count += 1
                    # Reduce learning rate
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    continue
                
                # Check gradients
                self.optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in self.policy.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    logger.warning("NaN gradients detected in PPO")
                    self.nan_count += 1
                    continue
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                
                self.optimizer.step()
                total_loss += loss.item()
        
        # Clear storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        return total_loss / (self.epochs * max(1, dataset_size // self.batch_size))
    
    def _compute_returns(self):
        """Compute discounted returns"""
        returns = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns
    
    def save(self, path):
        """Save model"""
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        """Load model"""
        self.policy.load_state_dict(torch.load(path))


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class TD3:
    """Twin Delayed DDPG"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256],
                 actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 buffer_size=1000000, batch_size=256, activation='relu',
                 noise_type='exploration'):
        """
        Initialize TD3
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: Hidden layer sizes
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            policy_noise: Noise added to target policy
            noise_clip: Range to clip target policy noise
            policy_delay: Frequency of delayed policy updates
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            activation: Activation function
            noise_type: 'exploration' or 'target_smoothing'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.noise_type = noise_type
        
        # Actor networks
        self.actor = MLP(state_dim, action_dim, hidden_sizes, activation)
        self.actor_target = MLP(state_dim, action_dim, hidden_sizes, activation)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Twin Critic networks
        self.critic1 = MLP(state_dim + action_dim, 1, hidden_sizes, activation)
        self.critic2 = MLP(state_dim + action_dim, 1, hidden_sizes, activation)
        self.critic1_target = MLP(state_dim + action_dim, 1, hidden_sizes, activation)
        self.critic2_target = MLP(state_dim + action_dim, 1, hidden_sizes, activation)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.train_steps = 0
        
    def select_action(self, state, noise=0.1):
        """Select action using current policy with exploration noise"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_tensor).squeeze(0).numpy()
            
            if self.noise_type == 'exploration' and noise > 0:
                action += np.random.normal(0, noise, size=self.action_dim)
            
            return np.clip(action, -1, 1)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform TD3 update"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            # Target policy smoothing
            noise = torch.randn_like(actions) * self.policy_noise
            if self.noise_type == 'target_smoothing':
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
            else:
                noise = torch.zeros_like(actions)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-1, 1)
            
            # Twin Q-targets
            target_q1 = self.critic1_target(torch.cat([next_states, next_actions], 1))
            target_q2 = self.critic2_target(torch.cat([next_states, next_actions], 1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q estimates
        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        loss = critic_loss.item()
        
        # Delayed policy updates
        if self.train_steps % self.policy_delay == 0:
            # Actor loss
            actor_loss = -self.critic1(torch.cat([states, self.actor(states)], 1)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            
            loss += actor_loss.item()
        
        self.train_steps += 1
        return loss
    
    def _soft_update(self, source, target):
        """Soft update of target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])


class SAC:
    """Soft Actor-Critic"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256],
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=1000000, batch_size=256, activation='relu',
                 auto_entropy_tuning=True):
        """
        Initialize SAC
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: Hidden layer sizes
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy regularization coefficient
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            activation: Activation function
            auto_entropy_tuning: Whether to automatically tune entropy
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Actor network
        self.actor = MLP(state_dim, action_dim * 2, hidden_sizes, activation)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Twin Critic networks
        self.critic1 = MLP(state_dim + action_dim, 1, hidden_sizes, activation)
        self.critic2 = MLP(state_dim + action_dim, 1, hidden_sizes, activation)
        self.critic1_target = MLP(state_dim + action_dim, 1, hidden_sizes, activation)
        self.critic2_target = MLP(state_dim + action_dim, 1, hidden_sizes, activation)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr
        )
        
        # Automatic entropy tuning
        if auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, log_std = self.actor(state_tensor).chunk(2, dim=-1)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.clamp(-20, 2).exp()
                dist = torch.distributions.Normal(mean, std)
                z = dist.rsample()
                action = torch.tanh(z)
            
            return action.squeeze(0).numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform SAC update"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            # Sample next actions
            next_mean, next_log_std = self.actor(next_states).chunk(2, dim=-1)
            next_std = next_log_std.clamp(-20, 2).exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_z = next_dist.rsample()
            next_actions = torch.tanh(next_z)
            next_log_probs = next_dist.log_prob(next_z) - torch.log(1 - next_actions.pow(2) + 1e-6)
            next_log_probs = next_log_probs.sum(dim=-1, keepdim=True)
            
            # Twin Q-targets
            target_q1 = self.critic1_target(torch.cat([next_states, next_actions], 1))
            target_q2 = self.critic2_target(torch.cat([next_states, next_actions], 1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q estimates
        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        mean, log_std = self.actor(states).chunk(2, dim=-1)
        std = log_std.clamp(-20, 2).exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        new_actions = torch.tanh(z)
        log_probs = dist.log_prob(z) - torch.log(1 - new_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        
        q1 = self.critic1(torch.cat([states, new_actions], 1))
        q2 = self.critic2(torch.cat([states, new_actions], 1))
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        return critic_loss.item() + actor_loss.item()
    
    def _soft_update(self, source, target):
        """Soft update of target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])


class Agent:
    """Agent for training RL algorithms"""
    
    def __init__(self, environment, policy):
        """
        Initialize Agent
        
        Args:
            environment: Environment wrapper
            policy: RL algorithm (PPO, TD3, or SAC)
        """
        self.env = environment
        self.policy = policy
        self.rewards_history = []
        
    def run_episode(self, train=True, render=False):
        """
        Run a single episode
        
        Args:
            train: Whether to train the policy
            render: Whether to render the environment
            
        Returns:
            Total episode reward
        """
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            if isinstance(self.policy, PPO):
                action, log_prob = self.policy.select_action(state)
                # Get value for PPO
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    _, _, value = self.policy.policy(state_tensor)
                    value = value.item()
            elif isinstance(self.policy, TD3):
                action = self.policy.select_action(state, noise=0.1 if train else 0)
            elif isinstance(self.policy, SAC):
                action = self.policy.select_action(state, deterministic=not train)
            else:
                raise ValueError(f"Unknown policy type: {type(self.policy)}")
            
            # Take step (clip action to prevent instability)
            action_clipped = np.clip(action, -1.0, 1.0)
            # Check for NaN/Inf in action
            if not np.all(np.isfinite(action_clipped)):
                logger.warning("NaN/Inf detected in action, using zero action")
                action_clipped = np.zeros_like(action)
            
            next_state, reward, done, truncated, info = self.env.step(action_clipped)
            episode_reward += reward
            
            # Store transition
            if train:
                if isinstance(self.policy, PPO):
                    self.policy.store_transition(state, action, reward, log_prob, value, done)
                else:
                    self.policy.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            
            if render:
                self.env.render()
        
        # Train at end of episode for PPO
        if train:
            if isinstance(self.policy, PPO):
                loss = self.policy.train_step()
            elif isinstance(self.policy, TD3) or isinstance(self.policy, SAC):
                # Train multiple times for off-policy methods
                loss = 0
                for _ in range(min(len(self.policy.memory) // self.policy.batch_size, 10)):
                    loss += self.policy.train_step()
            else:
                loss = 0
        
        self.rewards_history.append(episode_reward)
        return episode_reward
    
    def train(self, num_episodes, callback=None):
        """
        Train agent for multiple episodes
        
        Args:
            num_episodes: Number of episodes to train
            callback: Optional callback function called after each episode
            
        Returns:
            List of episode rewards
        """
        rewards = []
        
        for episode in range(num_episodes):
            reward = self.run_episode(train=True)
            rewards.append(reward)
            
            if callback:
                callback(episode, reward)
        
        return rewards
