"""
Pusher Environment and RL Algorithms
Implements DDPG, TD3, and SAC for Pusher-v5
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


def _activation_layer(name):
    name = (name or "relu").lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    return nn.ReLU()


class PusherEnvironment:
    """Wrapper for Pusher-v5 environment"""

    def __init__(self, reward_near_weight=0.5, reward_dist_weight=1.0,
                 reward_control_weight=0.1, render_mode=None):
        self.reward_near_weight = reward_near_weight
        self.reward_dist_weight = reward_dist_weight
        self.reward_control_weight = reward_control_weight
        self.render_mode = render_mode
        self.env = None
        self.reset_env()

    def reset_env(self):
        """Reset or create environment"""
        if self.env:
            self.env.close()
        self.env = gym.make(
            "Pusher-v5",
            reward_near_weight=self.reward_near_weight,
            reward_dist_weight=self.reward_dist_weight,
            reward_control_weight=self.reward_control_weight,
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

    def scale_action(self, action):
        """Scale action from [-1, 1] to env bounds"""
        low, high = self.get_action_bounds()
        action = np.clip(action, -1.0, 1.0)
        return low + (action + 1.0) * (high - low) / 2.0

    def render(self):
        """Render environment and return RGB array"""
        if self.env and self.render_mode == "rgb_array":
            return self.env.render()
        return None

    def close(self):
        """Close environment"""
        if self.env:
            self.env.close()


class MLPFeature(nn.Module):
    """MLP feature extractor"""

    def __init__(self, input_dim, hidden_dims, activation="relu"):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_activation_layer(activation))
        self.net = nn.Sequential(*layers)
        self.output_dim = dims[-1]

    def forward(self, x):
        return self.net(x)


class CNNFeature(nn.Module):
    """Simple 1D CNN feature extractor for vector inputs"""

    def __init__(self, input_dim, hidden_dims, activation="relu"):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        cnn_out = max(64, hidden_dims[0])
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            _activation_layer(activation),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            _activation_layer(activation)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * input_dim, cnn_out),
            _activation_layer(activation)
        )
        tail_layers = []
        dims = [cnn_out] + hidden_dims[1:]
        for i in range(len(dims) - 1):
            tail_layers.append(nn.Linear(dims[i], dims[i + 1]))
            tail_layers.append(_activation_layer(activation))
        self.tail = nn.Sequential(*tail_layers) if tail_layers else nn.Identity()
        self.output_dim = dims[-1] if dims else cnn_out

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.tail(x)
        return x


def build_feature_extractor(input_dim, hidden_dims, activation="relu", network_type="mlp"):
    if (network_type or "mlp").lower() == "cnn":
        return CNNFeature(input_dim, hidden_dims, activation)
    return MLPFeature(input_dim, hidden_dims, activation)


class ActorNetwork(nn.Module):
    """Deterministic actor network"""

    def __init__(self, state_dim, action_dim, hidden_dims, activation="relu", network_type="mlp"):
        super().__init__()
        self.feature = build_feature_extractor(state_dim, hidden_dims, activation, network_type)
        self.output_layer = nn.Linear(self.feature.output_dim, action_dim)

    def forward(self, x):
        x = self.feature(x)
        x = self.output_layer(x)
        return torch.tanh(x)


class CriticNetwork(nn.Module):
    """Q-value network"""

    def __init__(self, state_dim, action_dim, hidden_dims, activation="relu", network_type="mlp"):
        super().__init__()
        self.feature = build_feature_extractor(state_dim + action_dim, hidden_dims, activation, network_type)
        self.output_layer = nn.Linear(self.feature.output_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.feature(x)
        return self.output_layer(x)


class GaussianActor(nn.Module):
    """Gaussian policy for SAC"""

    def __init__(self, state_dim, action_dim, hidden_dims, activation="relu", network_type="mlp"):
        super().__init__()
        self.feature = build_feature_extractor(state_dim, hidden_dims, activation, network_type)
        self.mean = nn.Linear(self.feature.output_dim, action_dim)
        self.log_std = nn.Linear(self.feature.output_dim, action_dim)

    def forward(self, x):
        x = self.feature(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class DDPG:
    """Deep Deterministic Policy Gradient"""

    def __init__(self, state_dim, action_dim, hidden_dims=None, lr=3e-4,
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64,
                 exploration_noise=0.1, actor_activation="relu", critic_activation="relu",
                 network_type="mlp"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.network_type = network_type

        hidden_dims = hidden_dims or [256, 256]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims, actor_activation, network_type).to(self.device)
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_dims, actor_activation, network_type).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)
        self.lock = Lock()

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        if not evaluate:
            action = action + np.random.normal(0, self.exploration_noise, size=self.action_dim)
        return np.clip(action, -1.0, 1.0)

    def store_transition(self, state, action, reward, next_state, done):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
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

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (critic_loss.item() + actor_loss.item()) / 2.0


class TD3:
    """Twin Delayed Deep Deterministic Policy Gradient"""

    def __init__(self, state_dim, action_dim, hidden_dims=None, lr=3e-4,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, buffer_size=100000, batch_size=64,
                 exploration_noise=0.1, actor_activation="relu", critic_activation="relu",
                 network_type="mlp"):
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
        self.network_type = network_type

        hidden_dims = hidden_dims or [256, 256]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims, actor_activation, network_type).to(self.device)
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_dims, actor_activation, network_type).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)
        self.lock = Lock()

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        if not evaluate:
            action = action + np.random.normal(0, self.exploration_noise, size=self.action_dim)
        return np.clip(action, -1.0, 1.0)

    def store_transition(self, state, action, reward, next_state, done):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
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

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1, 1)

            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
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
        if self.total_steps % self.policy_delay == 0:
            new_actions = self.actor(states)
            actor_loss = -self.critic1(states, new_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (critic1_loss.item() + critic2_loss.item() + actor_loss.item()) / 3


class SAC:
    """Soft Actor-Critic"""

    def __init__(self, state_dim, action_dim, hidden_dims=None, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, buffer_size=100000,
                 batch_size=64, train_freq=4, gradient_steps=1,
                 actor_activation="relu", critic_activation="relu", network_type="mlp"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.total_steps = 0
        self.network_type = network_type

        hidden_dims = hidden_dims or [256, 256]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = GaussianActor(state_dim, action_dim, hidden_dims, actor_activation, network_type).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dims, critic_activation, network_type).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)
        self.lock = Lock()

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = log_std.exp()
            if evaluate:
                action = torch.tanh(mean)
            else:
                dist = torch.distributions.Normal(mean, std)
                z = dist.sample()
                action = torch.tanh(z)
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
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

        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_z = next_dist.rsample()
            next_actions = torch.tanh(next_z)
            next_log_probs = next_dist.log_prob(next_z).sum(dim=-1, keepdim=True)
            next_log_probs -= torch.sum(torch.log(1 - next_actions.pow(2) + 1e-6), dim=-1, keepdim=True)

            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
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

        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        new_actions = torch.tanh(z)
        log_probs = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_probs -= torch.sum(torch.log(1 - new_actions.pow(2) + 1e-6), dim=-1, keepdim=True)

        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

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
        self.current_episode_reward = 0.0

    def set_policy(self, policy):
        self.policy = policy

    def run_episode(self, render=False, train=True, render_callback=None, max_steps=1000):
        state = self.environment.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            if render and render_callback and steps % 2 == 0:
                frame = self.environment.render()
                if frame is not None:
                    render_callback(frame)

            action = self.policy.select_action(state)
            scaled_action = self.environment.scale_action(action)
            next_state, reward, done, _ = self.environment.step(scaled_action)

            if train:
                self.policy.store_transition(state, action, reward, next_state, done)
                if hasattr(self.policy, "train_step"):
                    if isinstance(self.policy, TD3) or isinstance(self.policy, DDPG):
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

        self.episode_rewards.append(episode_reward)
        self.current_episode_reward = episode_reward
        return episode_reward, steps

    def get_returns(self):
        return self.episode_rewards.copy()

    def reset_stats(self):
        self.episode_rewards = []
        self.current_episode_reward = 0.0
