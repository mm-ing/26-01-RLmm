from __future__ import annotations

import random
import time
from dataclasses import dataclass
from threading import Event
from typing import Callable, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class LunarLanderCEnv:
    def __init__(
        self,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        continuous: bool = True,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        self.continuous = bool(continuous)
        
        env_name = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
        
        self.env = gym.make(
            env_name,
            render_mode=render_mode,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )
        
        if seed is not None:
            self.env.reset(seed=seed)
        
        # For continuous env
        if continuous:
            if not isinstance(self.env.action_space, spaces.Box):
                raise ValueError("Continuous LunarLander action space must be Box.")
            self.action_dim = int(self.env.action_space.shape[0])
            self.action_low = self.env.action_space.low
            self.action_high = self.env.action_space.high
        else:
            if not isinstance(self.env.action_space, spaces.Discrete):
                raise ValueError("Discrete LunarLander action space must be Discrete.")
            self.action_dim = int(self.env.action_space.n)
            self.action_low = None
            self.action_high = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self) -> np.ndarray:
        obs, _info = self.env.reset()
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        if self.continuous:
            # Clip action to valid range
            action = np.clip(action, self.action_low, self.action_high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ============================================================================
# Replay Buffer for Rainbow
# ============================================================================
class ReplayBuffer:
    def __init__(self, size: int, state_dim: int, action_dim: int):
        self.size = size
        self.pos = 0
        self.count = 0
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.action_indices = np.zeros(size, dtype=np.int64)  # Store discrete action indices
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

    def add(self, state, action, reward, next_state, done, action_idx=0):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.action_indices[self.pos] = action_idx
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size: int):
        indices = np.random.choice(self.count, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.action_indices[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.count


# ============================================================================
# Prioritized Replay Buffer for Rainbow
# ============================================================================
class PrioritizedReplayBuffer:
    def __init__(self, size: int, state_dim: int, action_dim: int, alpha: float = 0.6):
        self.size = size
        self.alpha = alpha
        self.pos = 0
        self.count = 0
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.action_indices = np.zeros(size, dtype=np.int64)  # Store discrete action indices
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.priorities = np.zeros(size, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done, action_idx=0):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.action_indices[self.pos] = action_idx
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size: int, beta: float = 0.4):
        if self.count == 0:
            raise ValueError("Buffer is empty")
        
        priorities = self.priorities[:self.count] ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(self.count, size=batch_size, p=probs, replace=False)
        
        weights = (self.count * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        return (
            self.states[indices],
            self.actions[indices],
            self.action_indices[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights,
        )

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self):
        return self.count


# ============================================================================
# Neural Networks
# ============================================================================
class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""
    def __init__(self, in_features: int, out_features: int, sigma: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """Rainbow DQN network for continuous action space (discretized)."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 256,
        activation: str = "ReLU",
        atoms: int = 51,
        vmin: float = -300.0,
        vmax: float = 300.0,
        noisy_sigma: float = 0.5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax
        
        # Distributional RL support
        self.register_buffer('support', torch.linspace(vmin, vmax, atoms))
        self.delta_z = (vmax - vmin) / (atoms - 1)
        
        # Activation function
        if activation == "ReLU":
            act_fn = nn.ReLU
        elif activation == "LeakyReLU":
            act_fn = nn.LeakyReLU
        elif activation == "Tanh":
            act_fn = nn.Tanh
        elif activation == "ELU":
            act_fn = nn.ELU
        elif activation == "GELU":
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU
        
        # Feature extraction (shared)
        layers = []
        in_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(NoisyLinear(in_dim, hidden_units, noisy_sigma))
            layers.append(act_fn())
            in_dim = hidden_units
        self.features = nn.Sequential(*layers)
        
        # Dueling architecture: Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_units, hidden_units // 2, noisy_sigma),
            act_fn(),
            NoisyLinear(hidden_units // 2, atoms, noisy_sigma),
        )
        
        # Dueling architecture: Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_units, hidden_units // 2, noisy_sigma),
            act_fn(),
            NoisyLinear(hidden_units // 2, action_dim * atoms, noisy_sigma),
        )

    def forward(self, state):
        features = self.features(state)
        value = self.value_stream(features).view(-1, 1, self.atoms)
        advantage = self.advantage_stream(features).view(-1, self.action_dim, self.atoms)
        
        # Dueling aggregation
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_q_values(self, state):
        q_dist = self.forward(state)
        q_values = (q_dist * self.support).sum(dim=-1)
        return q_values


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for continuous control."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 256,
        activation: str = "ReLU",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "ReLU":
            act_fn = nn.ReLU
        elif activation == "LeakyReLU":
            act_fn = nn.LeakyReLU
        elif activation == "Tanh":
            act_fn = nn.Tanh
        elif activation == "ELU":
            act_fn = nn.ELU
        elif activation == "GELU":
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU
        
        # Shared feature extraction
        layers = []
        in_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(act_fn())
            in_dim = hidden_units
        self.features = nn.Sequential(*layers)
        
        # Actor: outputs mean and log_std for Gaussian policy
        self.actor_mean = nn.Linear(hidden_units, action_dim)
        self.actor_log_std = nn.Linear(hidden_units, action_dim)
        
        # Critic: outputs state value
        self.critic = nn.Linear(hidden_units, 1)
        
        # Initialize weights for better learning
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for better convergence."""
        # Initialize feature layers with orthogonal initialization
        for module in self.features.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Initialize actor mean with small weights
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
        # Initialize actor log_std to give std around 0.5 for better exploration
        nn.init.constant_(self.actor_log_std.weight, 0.0)
        nn.init.constant_(self.actor_log_std.bias, -0.5)
        
        # Initialize critic
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, state):
        features = self.features(state)
        
        # Actor outputs
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        
        # Critic output
        value = self.critic(features)
        
        return mean, log_std, value

    def get_action(self, state, deterministic=False):
        mean, log_std, value = self.forward(state)
        
        if deterministic:
            action_tanh = torch.tanh(mean)
            return action_tanh, value, torch.zeros(state.shape[0], 1, device=state.device)
        
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        action_tanh = torch.tanh(action)
        
        # Calculate log probability with correction for tanh
        log_prob = dist.log_prob(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action_tanh, value, log_prob

    def evaluate_action(self, state, action):
        """Evaluate actions for PPO."""
        mean, log_std, value = self.forward(state)
        std = log_std.exp()
        
        # Convert tanh action back to pre-tanh space for evaluation
        action_raw = torch.atanh(torch.clamp(action, -0.9999, 0.9999))
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return value, log_prob, entropy


# ============================================================================
# Base Policy
# ============================================================================
@dataclass
class StepResult:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class BasePolicy:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def update(self, step_result: StepResult):
        raise NotImplementedError

    def reset(self):
        self.steps = 0


# ============================================================================
# Rainbow Policy (from existing implementation)
# ============================================================================
class RainbowPolicy(BasePolicy):
    """Rainbow DQN for continuous action space (discretized)."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.0005,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.001,
        hidden_layers: int = 2,
        hidden_units: int = 256,
        batch_size: int = 64,
        replay_size: int = 100000,
        activation: str = "ReLU",
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
        target_update: int = 100,
        priority_alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100000,
        atoms: int = 51,
        vmin: float = -300.0,
        vmax: float = 300.0,
        noisy_sigma: float = 0.5,
        n_step: int = 3,
    ):
        super().__init__(state_dim, action_dim)
        
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.target_update = target_update
        self.priority_alpha = priority_alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax
        self.n_step = n_step
        
        # Discretize continuous action space for Rainbow
        # For LunarLander continuous: 2D action (main engine, side engines)
        # Discretize each dimension into bins
        self.action_bins = 5
        self.total_discrete_actions = self.action_bins ** action_dim
        
        # Networks
        self.network = RainbowNetwork(
            state_dim, self.total_discrete_actions, hidden_layers, hidden_units,
            activation, atoms, vmin, vmax, noisy_sigma
        ).to(self.device)
        
        self.target_network = RainbowNetwork(
            state_dim, self.total_discrete_actions, hidden_layers, hidden_units,
            activation, atoms, vmin, vmax, noisy_sigma
        ).to(self.device)
        
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=alpha)
        self.replay_buffer = PrioritizedReplayBuffer(replay_size, state_dim, action_dim, priority_alpha)
        
        # N-step buffer
        self.n_step_buffer = []
        
        # Support for distributional RL
        self.support = torch.linspace(vmin, vmax, atoms).to(self.device)
        self.delta_z = (vmax - vmin) / (atoms - 1)

    def _discrete_to_continuous(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to continuous action."""
        action = np.zeros(self.action_dim, dtype=np.float32)
        for i in range(self.action_dim):
            bin_idx = (action_idx // (self.action_bins ** i)) % self.action_bins
            # Map bin to [-1, 1]
            action[i] = -1.0 + (bin_idx / (self.action_bins - 1)) * 2.0
        return action

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Noisy networks handle exploration, but we can still use epsilon for diversity
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.eps_decay * self.steps)
        
        if not deterministic and random.random() < epsilon:
            action_idx = random.randint(0, self.total_discrete_actions - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.network.get_q_values(state_t)
                action_idx = q_values.argmax(dim=1).item()
        
        # Store for later use in update
        self.last_action_idx = action_idx
        return self._discrete_to_continuous(action_idx)

    def update(self, step_result: StepResult):
        self.steps += 1
        
        # Add to n-step buffer
        self.n_step_buffer.append(step_result)
        
        # Calculate n-step return
        if len(self.n_step_buffer) >= self.n_step or step_result.done:
            n_step_return = 0.0
            for i, transition in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * transition.reward
            
            first_transition = self.n_step_buffer[0]
            last_transition = self.n_step_buffer[-1]
            
            # Convert continuous action back to discrete index for storage
            # Use the stored action index from select_action
            action_idx = getattr(self, 'last_action_idx', 0)
            
            self.replay_buffer.add(
                first_transition.state,
                first_transition.action,
                n_step_return,
                last_transition.next_state,
                last_transition.done,
                action_idx,
            )
            
            self.n_step_buffer.pop(0)
        
        if step_result.done:
            self.n_step_buffer.clear()
        
        # Training
        if len(self.replay_buffer) >= max(self.warmup_steps, self.batch_size):
            self._train()
        
        # Update target network
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def _train(self):
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.steps / self.beta_steps)
        
        states, actions, action_indices, rewards, next_states, dones, buffer_indices, weights = self.replay_buffer.sample(
            self.batch_size, beta
        )
        
        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        action_indices_t = torch.LongTensor(action_indices).to(self.device)
        
        # Get current Q distribution
        self.network.reset_noise()
        current_q_dist = self.network(states_t)
        
        # Get next Q distribution (Double DQN)
        with torch.no_grad():
            self.target_network.reset_noise()
            next_q_values = self.network.get_q_values(next_states_t)
            next_actions = next_q_values.argmax(dim=1)
            next_q_dist = self.target_network(next_states_t)
            next_q_dist = next_q_dist[range(self.batch_size), next_actions]
            
            # Distributional Bellman update
            t_z = rewards_t.unsqueeze(1) + (1 - dones_t.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            t_z = t_z.clamp(self.vmin, self.vmax)
            
            b = (t_z - self.vmin) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.atoms).to(self.device)
            
            proj_dist = torch.zeros(next_q_dist.size()).to(self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_q_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_q_dist * (b - l.float())).view(-1))
        
        # Use the actual action indices that were taken
        current_q_dist = current_q_dist[range(self.batch_size), action_indices_t]
        
        # Calculate loss
        loss = -(proj_dist * current_q_dist.clamp(min=1e-8).log()).sum(dim=1)
        prio_loss = loss.clone()
        loss = (loss * weights_t).mean()
        
        # Update priorities
        with torch.no_grad():
            priorities = prio_loss.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(buffer_indices, priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

    def reset(self):
        super().reset()
        self.n_step_buffer.clear()


# ============================================================================
# A2C Policy (Advantage Actor-Critic)
# ============================================================================
class A2CPolicy(BasePolicy):
    """A2C for continuous control."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.0003,
        gamma: float = 0.99,
        hidden_layers: int = 2,
        hidden_units: int = 256,
        activation: str = "ReLU",
        grad_clip: float = 0.5,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 5,
    ):
        super().__init__(state_dim, action_dim)
        
        self.alpha = alpha
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_layers, hidden_units, activation
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=alpha)
        
        # Storage for n-step updates
        self.buffer = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.network.get_action(state_t, deterministic)
        
        return action.cpu().numpy()[0]

    def update(self, step_result: StepResult):
        self.steps += 1
        self.buffer.append(step_result)
        
        # Update every n_steps or at end of episode
        if len(self.buffer) >= self.n_steps or step_result.done:
            if len(self.buffer) > 0:
                self._train()
            self.buffer.clear()

    def _train(self):
        if len(self.buffer) == 0:
            return
        
        # Prepare data
        states = np.array([t.state for t in self.buffer])
        actions = np.array([t.action for t in self.buffer])
        rewards = np.array([t.reward for t in self.buffer])
        next_states = np.array([t.next_state for t in self.buffer])
        dones = np.array([t.done for t in self.buffer])
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        
        with torch.no_grad():
            _, _, values = self.network.forward(states_t)
            values = values.squeeze(-1).cpu().numpy()
            
            last_value = 0
            if not self.buffer[-1].done:
                next_state_t = torch.FloatTensor(self.buffer[-1].next_state).unsqueeze(0).to(self.device)
                _, _, last_value_t = self.network.forward(next_state_t)
                last_value = last_value_t.item()
            
            R = last_value
            for i in reversed(range(len(self.buffer))):
                R = rewards[i] + self.gamma * R * (1 - dones[i])
                returns.insert(0, R)
                advantages.insert(0, R - values[i])
        
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_np = np.array(advantages, dtype=np.float32)
        # ALWAYS normalize advantages to prevent gradient explosion
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)
        advantages_t = torch.FloatTensor(advantages_np).to(self.device)
        
        # Single forward pass to get both actor and critic outputs
        mean, log_std, values = self.network.forward(states_t)
        values = values.squeeze(-1)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        # Reverse tanh transformation more safely
        actions_clamped = torch.clamp(actions_t, -0.99999, 0.99999)
        actions_raw = torch.atanh(actions_clamped)
        
        log_probs = dist.log_prob(actions_raw)
        # Apply tanh correction
        log_probs -= torch.log(1 - actions_clamped.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Losses
        policy_loss = -(log_probs * advantages_t).mean()
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

    def reset(self):
        super().reset()
        self.buffer.clear()


# ============================================================================
# TRPO Policy (Trust Region Policy Optimization)
# ============================================================================
class TRPOPolicy(BasePolicy):
    """TRPO for continuous control (simplified version)."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        hidden_layers: int = 2,
        hidden_units: int = 256,
        activation: str = "ReLU",
        max_kl: float = 0.01,
        damping: float = 0.1,
        value_lr: float = 0.001,
        n_steps: int = 2048,
        gae_lambda: float = 0.95,
    ):
        super().__init__(state_dim, action_dim)
        
        self.gamma = gamma
        self.max_kl = max_kl
        self.damping = damping
        self.value_lr = value_lr
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.batch_size = 32  # Minimum batch size for training
        
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_layers, hidden_units, activation
        ).to(self.device)
        
        # Separate optimizer for value function
        self.value_optimizer = optim.Adam(self.network.critic.parameters(), lr=value_lr)
        
        # Storage
        self.buffer = []
        self.episode_count = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.network.get_action(state_t, deterministic)
        
        return action.cpu().numpy()[0]

    def update(self, step_result: StepResult):
        self.steps += 1
        self.buffer.append(step_result)
        
        # Train when buffer reaches n_steps or at episode end
        if step_result.done:
            self.episode_count += 1
        
        if len(self.buffer) >= self.n_steps or (step_result.done and len(self.buffer) > self.batch_size):
            if len(self.buffer) > 0:
                self._train()
            self.buffer.clear()

    def _train(self):
        # Prepare data
        states = np.array([t.state for t in self.buffer])
        actions = np.array([t.action for t in self.buffer])
        rewards = np.array([t.reward for t in self.buffer])
        dones = np.array([t.done for t in self.buffer])
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        
        # Compute GAE
        with torch.no_grad():
            _, _, values = self.network.forward(states_t)
            values_np = values.squeeze(-1).cpu().numpy()
            
            advantages = np.zeros(len(self.buffer), dtype=np.float32)
            returns = np.zeros(len(self.buffer), dtype=np.float32)
            
            last_value = 0
            if not self.buffer[-1].done:
                next_state_t = torch.FloatTensor(self.buffer[-1].next_state).unsqueeze(0).to(self.device)
                _, _, last_value_t = self.network.forward(next_state_t)
                last_value = last_value_t.item()
            
            last_gae = 0
            for i in reversed(range(len(self.buffer))):
                if dones[i]:
                    next_value = 0
                elif i == len(self.buffer) - 1:
                    next_value = last_value
                else:
                    next_value = values_np[i + 1]
                
                delta = rewards[i] + self.gamma * next_value - values_np[i]
                last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * last_gae
                advantages[i] = last_gae
                returns[i] = advantages[i] + values_np[i]
            
            # ALWAYS normalize advantages to prevent gradient explosion
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Get old policy distribution
        with torch.no_grad():
            mean_old, log_std_old, _ = self.network.forward(states_t)
            actions_raw = torch.atanh(torch.clamp(actions_t, -0.9999, 0.9999))
            std_old = log_std_old.exp()
            dist_old = Normal(mean_old, std_old)
            log_probs_old = dist_old.log_prob(actions_raw)
            log_probs_old -= torch.log(1 - actions_t.pow(2) + 1e-6)
            log_probs_old = log_probs_old.sum(dim=-1)
        
        # TRPO update (simplified - using line search instead of conjugate gradient)
        def get_loss_and_kl():
            mean, log_std, _ = self.network.forward(states_t)
            std = log_std.exp()
            dist = Normal(mean, std)
            
            log_probs = dist.log_prob(actions_raw)
            log_probs -= torch.log(1 - actions_t.pow(2) + 1e-6)
            log_probs = log_probs.sum(dim=-1)
            
            ratio = torch.exp(log_probs - log_probs_old)
            loss = -(ratio * advantages_t).mean()
            
            # KL divergence
            kl = (log_probs_old - log_probs).mean()
            
            return loss, kl
        
        # Create optimizer for policy if it doesn't exist
        if not hasattr(self, 'policy_optimizer'):
            self.policy_optimizer = optim.Adam(
                list(self.network.features.parameters()) +
                list(self.network.actor_mean.parameters()) +
                list(self.network.actor_log_std.parameters()),
                lr=0.0003
            )
        
        # Simple line search - update policy only if KL is reasonable
        loss, kl = get_loss_and_kl()
        if kl.item() < self.max_kl:
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.network.features.parameters()) +
                list(self.network.actor_mean.parameters()) +
                list(self.network.actor_log_std.parameters()),
                0.5
            )
            self.policy_optimizer.step()
        
        # Update value function (fewer iterations to prevent overfitting)
        for _ in range(3):
            _, _, values = self.network.forward(states_t)
            value_loss = F.mse_loss(values.squeeze(-1), returns_t)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), 0.5)
            self.value_optimizer.step()

    def reset(self):
        super().reset()
        self.buffer.clear()


# ============================================================================
# PPO Policy (Proximal Policy Optimization)
# ============================================================================
class PPOPolicy(BasePolicy):
    """PPO for continuous control."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.0003,
        gamma: float = 0.99,
        hidden_layers: int = 2,
        hidden_units: int = 256,
        activation: str = "ReLU",
        grad_clip: float = 0.5,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        gae_lambda: float = 0.95,
        update_frequency: int = 1,  # Update every N episodes
    ):
        super().__init__(state_dim, action_dim)
        
        self.alpha = alpha
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.update_frequency = update_frequency
        
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_layers, hidden_units, activation
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=alpha)
        
        # Storage
        self.buffer = []
        self.episode_count = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.network.get_action(state_t, deterministic)
        
        return action.cpu().numpy()[0]

    def update(self, step_result: StepResult):
        self.steps += 1
        self.buffer.append(step_result)
        
        # Update every n_steps or at end of episode
        if len(self.buffer) >= self.n_steps or step_result.done:
            if len(self.buffer) > 0:
                self._train()
            self.buffer.clear()

    def _train(self):
        # Skip if buffer is too small
        if len(self.buffer) < self.batch_size:
            return
            
        # Prepare data
        states = np.array([t.state for t in self.buffer])
        actions = np.array([t.action for t in self.buffer])
        rewards = np.array([t.reward for t in self.buffer])
        dones = np.array([t.done for t in self.buffer])
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        
        # Compute GAE
        with torch.no_grad():
            _, _, values = self.network.forward(states_t)
            values_np = values.squeeze(-1).cpu().numpy()
            
            advantages = np.zeros(len(self.buffer), dtype=np.float32)
            returns = np.zeros(len(self.buffer), dtype=np.float32)
            
            last_value = 0
            if not self.buffer[-1].done:
                next_state_t = torch.FloatTensor(self.buffer[-1].next_state).unsqueeze(0).to(self.device)
                _, _, last_value_t = self.network.forward(next_state_t)
                last_value = last_value_t.item()
            
            last_gae = 0
            for i in reversed(range(len(self.buffer))):
                if dones[i]:
                    next_value = 0
                elif i == len(self.buffer) - 1:
                    next_value = last_value
                else:
                    next_value = values_np[i + 1]
                
                delta = rewards[i] + self.gamma * next_value - values_np[i]
                last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * last_gae
                advantages[i] = last_gae
                returns[i] = advantages[i] + values_np[i]
            
            # ALWAYS normalize advantages to prevent gradient explosion
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get old log probabilities
            mean_old, log_std_old, _ = self.network.forward(states_t)
            std_old = log_std_old.exp()
            dist_old = Normal(mean_old, std_old)
            # Reverse tanh transformation more safely
            actions_clamped = torch.clamp(actions_t, -0.99999, 0.99999)
            actions_raw = torch.atanh(actions_clamped)
            log_probs_old = dist_old.log_prob(actions_raw)
            log_probs_old -= torch.log(1 - actions_clamped.pow(2) + 1e-6)
            log_probs_old = log_probs_old.sum(dim=-1, keepdim=True)
        
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # PPO epochs
        for _ in range(self.n_epochs):
            # Create mini-batches
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.buffer), self.batch_size):
                end = min(start + self.batch_size, len(self.buffer))
                batch_indices = indices[start:end]
                
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                
                # Forward pass
                values, log_probs, entropy = self.network.evaluate_action(batch_states, batch_actions)
                
                # Policy loss (clipped)
                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                self.optimizer.step()

    def reset(self):
        super().reset()
        self.buffer.clear()
        self.episode_count = 0


# ============================================================================
# Agent
# ============================================================================
class Agent:
    def __init__(self, env: LunarLanderCEnv, policy: BasePolicy):
        self.env = env
        self.policy = policy

    def run_episode(
        self,
        max_steps: int = 1000,
        render_callback: Optional[Callable] = None,
        step_delay: float = 0.0,
        stop_event: Optional[Event] = None,
        deterministic: bool = False,
    ) -> Tuple[float, int]:
        state = self.env.reset()
        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            if stop_event and stop_event.is_set():
                break

            action = self.policy.select_action(state, deterministic)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            steps += 1

            if not deterministic:
                step_result = StepResult(state, action, reward, next_state, done)
                self.policy.update(step_result)

            if render_callback:
                render_callback()

            if step_delay > 0:
                time.sleep(step_delay)

            state = next_state

            if done:
                break

        return total_reward, steps
