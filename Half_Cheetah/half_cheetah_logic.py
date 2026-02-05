"""
HalfCheetah Environment and RL Algorithms
Implements PPO, TD3, and SAC algorithms for HalfCheetah-v5
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


class HalfCheetahEnvironment:
    """Wrapper for HalfCheetah-v5 environment"""
    
    def __init__(self, forward_reward_weight=1.0, ctrl_cost_weight=0.1, 
                 reset_noise_scale=0.1, exclude_current_positions_from_observation=True,
                 render_mode=None):
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.reset_noise_scale = reset_noise_scale
        self.exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self.render_mode = render_mode
        self.env = None
        self.reset_env()
        
    def reset_env(self):
        """Reset or create environment"""
        if self.env:
            self.env.close()
        self.env = gym.make(
            'HalfCheetah-v5',
            forward_reward_weight=self.forward_reward_weight,
            ctrl_cost_weight=self.ctrl_cost_weight,
            reset_noise_scale=self.reset_noise_scale,
            exclude_current_positions_from_observation=self.exclude_current_positions_from_observation,
            render_mode=self.render_mode
        )
        
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
        
        if activation.lower() == 'relu':
            self.activation = F.relu
        elif activation.lower() == 'tanh':
            self.activation = torch.tanh
        elif activation.lower() == 'leaky_relu':
            self.activation = F.leaky_relu
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
            layer = nn.Linear(dims[i], dims[i+1])
            # Xavier initialization for stability
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
            self.actor_layers.append(layer)
        
        self.mean_layer = nn.Linear(dims[-1], action_dim)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        
        self.log_std_layer = nn.Linear(dims[-1], action_dim)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0.0)
        
        # Critic (value) network
        self.critic_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1])
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
            self.critic_layers.append(layer)
        
        self.value_layer = nn.Linear(dims[-1], 1)
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        nn.init.constant_(self.value_layer.bias, 0.0)
        
    def forward(self, x):
        # Actor
        actor_x = x
        for layer in self.actor_layers:
            actor_x = F.relu(layer(actor_x))
        
        mean = self.mean_layer(actor_x)
        log_std = self.log_std_layer(actor_x)
        log_std = torch.clamp(log_std, -20, 2)
        
        # Critic
        critic_x = x
        for layer in self.critic_layers:
            critic_x = F.relu(layer(critic_x))
        
        value = self.value_layer(critic_x)
        
        return mean, log_std, value


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
        self.base_lr = lr
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.lock = Lock()
        self.episode_count = 0
        self.nan_count = 0
        
    def select_action(self, state):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std, _ = self.network(state_tensor)
            
            # Check for NaN
            if torch.isnan(mean).any() or torch.isnan(log_std).any():
                print(f"Warning: NaN detected in policy output (episode {self.episode_count}), using random action")
                self.nan_count += 1
                # Return random action without storing for training
                return np.random.uniform(-1, 1, self.action_dim)
            
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp() + 1e-6
            
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
        
        self.episode_count += 1
        
        # Adaptive learning rate (reduce if NaN occurred recently)
        if self.nan_count > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * 0.1
        
        # Compute returns with normalization
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Multiple epochs of updates
        total_loss = 0
        for _ in range(self.update_epochs):
            mean, log_std, values = self.network(states)
            
            # Clamp log_std to prevent extreme values
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            
            # Add small epsilon for numerical stability
            std = std + 1e-6
            
            # Check for NaN values
            if torch.isnan(mean).any() or torch.isnan(std).any():
                print("Warning: NaN detected in network output, skipping update")
                self.nan_count += 1
                return 0.0
            
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Compute advantages
            with torch.no_grad():
                _, _, baseline_values = self.network(states)
            advantages = returns - baseline_values.squeeze()
            
            # Normalize advantages for stability
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0.1, 10.0)  # Prevent extreme ratios
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            value_pred_clipped = baseline_values.squeeze() + torch.clamp(
                values.squeeze() - baseline_values.squeeze(),
                -self.clip_epsilon, self.clip_epsilon
            )
            value_loss1 = F.mse_loss(values.squeeze(), returns)
            value_loss2 = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print("Warning: NaN detected in loss, skipping update")
                self.nan_count += 1
                return 0.0
            
            # Update with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            
            # Check gradients for NaN
            has_nan_grad = False
            for param in self.network.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print("Warning: NaN in gradients, skipping update")
                self.nan_count += 1
                return 0.0
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Reset NaN counter if training was successful
        if self.nan_count > 0:
            self.nan_count = max(0, self.nan_count - 1)
        
        return total_loss / self.update_epochs


class TD3:
    """Twin Delayed Deep Deterministic Policy Gradient"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], lr=3e-4,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, buffer_size=100000, batch_size=64, exploration_noise=0.1,
                 actor_activation='relu', critic_activation='relu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.total_steps = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network (deterministic policy)
        self.actor = MLP(state_dim, action_dim, hidden_dims, actor_activation).to(self.device)
        self.target_actor = MLP(state_dim, action_dim, hidden_dims, actor_activation).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # Twin critic networks
        self.critic1 = MLP(state_dim + action_dim, 1, hidden_dims, critic_activation).to(self.device)
        self.critic2 = MLP(state_dim + action_dim, 1, hidden_dims, critic_activation).to(self.device)
        
        # Target critic networks
        self.target_critic1 = MLP(state_dim + action_dim, 1, hidden_dims, critic_activation).to(self.device)
        self.target_critic2 = MLP(state_dim + action_dim, 1, hidden_dims, critic_activation).to(self.device)
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
            action = self.actor(state_tensor)
            action = torch.tanh(action)  # Bound action to [-1, 1]
            
            if not evaluate:
                # Add exploration noise
                noise = torch.randn_like(action) * self.exploration_noise
                action = action + noise
                action = torch.clamp(action, -1, 1)
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        self.total_steps += 1
        
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
            # Target policy smoothing: add noise to target actions
            next_actions = torch.tanh(self.target_actor(next_states))
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + noise
            next_actions = torch.clamp(next_actions, -1, 1)
            
            next_state_actions = torch.cat([next_states, next_actions], dim=1)
            target_q1 = self.target_critic1(next_state_actions)
            target_q2 = self.target_critic2(next_state_actions)
            target_q = torch.min(target_q1, target_q2)
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
        
        actor_loss = torch.tensor(0.0)
        
        # Delayed policy updates
        if self.total_steps % self.policy_delay == 0:
            # Freeze critic networks during actor update
            for param in self.critic1.parameters():
                param.requires_grad = False
            
            # Update actor
            new_actions = torch.tanh(self.actor(states))
            new_state_actions = torch.cat([states, new_actions], dim=1)
            actor_loss = -self.critic1(new_state_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Unfreeze critic networks
            for param in self.critic1.parameters():
                param.requires_grad = True
            
            # Soft update target networks
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return (critic1_loss.item() + critic2_loss.item() + actor_loss.item()) / 3


class SAC:
    """Soft Actor-Critic"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, buffer_size=100000, batch_size=64,
                 train_freq=4, gradient_steps=1, actor_activation='relu', critic_activation='relu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.total_steps = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network (uses ActorCritic which has fixed relu - this is for consistency)
        self.actor = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Critic networks (twin Q-functions)
        self.critic1 = MLP(state_dim + action_dim, 1, hidden_dims, critic_activation).to(self.device)
        self.critic2 = MLP(state_dim + action_dim, 1, hidden_dims, critic_activation).to(self.device)
        
        # Target critic networks
        self.target_critic1 = MLP(state_dim + action_dim, 1, hidden_dims, critic_activation).to(self.device)
        self.target_critic2 = MLP(state_dim + action_dim, 1, hidden_dims, critic_activation).to(self.device)
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
        max_steps = 1000  # HalfCheetah default episode length
        
        while not done and steps < max_steps:
            # Render if callback provided
            if render and render_callback and steps % 2 == 0:
                frame = self.environment.render()
                if frame is not None:
                    render_callback(frame)
            
            action = self.policy.select_action(state)
            next_state, reward, done, _ = self.environment.step(action)
            
            if train:
                self.policy.store_transition(state, action, reward, next_state, done)
                
                # Train step for off-policy algorithms
                if hasattr(self.policy, 'train_step'):
                    if isinstance(self.policy, TD3):
                        # TD3 trains every step after enough samples
                        if len(self.policy.memory) >= self.policy.batch_size * 2:
                            self.policy.train_step()
                    elif isinstance(self.policy, SAC):
                        self.policy.total_steps += 1
                        if (len(self.policy.memory) >= self.policy.batch_size * 2 and 
                            self.policy.total_steps % self.policy.train_freq == 0):
                            for _ in range(self.policy.gradient_steps):
                                self.policy.train_step()
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Train step for on-policy algorithms
        if train and hasattr(self.policy, 'train_step'):
            if isinstance(self.policy, PPO):
                self.policy.train_step()
        
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
