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


class LunarLanderEnv:
    def __init__(
        self,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        continuous: bool = False,
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
        
        # For continuous env, discretize actions for DQN-based methods
        if continuous:
            if not isinstance(self.env.action_space, spaces.Box):
                raise ValueError("Continuous LunarLander action space must be Box.")
            # Continuous has 2D action space: main engine and side engines
            self.action_bins = 5  # Discretize each dimension into 5 bins
            low = self.env.action_space.low
            high = self.env.action_space.high
            self.action_values = []
            for i in range(len(low)):
                self.action_values.append(np.linspace(low[i], high[i], self.action_bins, dtype=np.float32))
            self.total_actions = self.action_bins ** len(low)  # 5^2 = 25 actions
        else:
            if not isinstance(self.env.action_space, spaces.Discrete):
                raise ValueError("Discrete LunarLander action space must be Discrete.")
            self.total_actions = int(self.env.action_space.n)  # 4 actions
            self.action_values = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def action_from_index(self, action_idx: int):
        """Convert action index to environment action."""
        if self.action_values is None:
            # Discrete environment
            return int(np.clip(action_idx, 0, self.total_actions - 1))
        else:
            # Continuous environment with discretized actions
            idx = int(np.clip(action_idx, 0, self.total_actions - 1))
            # Map linear index to multi-dimensional action
            action = []
            for i in range(len(self.action_values)):
                bin_idx = (idx // (self.action_bins ** i)) % self.action_bins
                action.append(self.action_values[i][bin_idx])
            return np.array(action, dtype=np.float32)

    def reset(self) -> np.ndarray:
        obs, _info = self.env.reset()
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        if self.action_values is None:
            obs, reward, terminated, truncated, info = self.env.step(int(action))
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


@dataclass
class EpsilonSchedule:
    eps_start: float
    eps_end: float
    eps_decay: float

    def epsilon_for(self, episode_idx: int) -> float:
        eps = self.eps_start - self.eps_decay * episode_idx
        return max(self.eps_end, eps)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.pos = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        item = (state, int(action), float(reward), next_state, bool(done))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.data)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.data: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, priority: Optional[float] = None):
        item = (state, int(action), float(reward), next_state, bool(done))
        max_prio = float(self.priorities.max()) if self.data else 1.0
        prio = float(priority) if priority is not None else max_prio
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.data) == 0:
            raise ValueError("Cannot sample from empty replay buffer.")
        prios = self.priorities[: len(self.data)]
        prios = np.maximum(prios, 1e-6)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.data), batch_size, p=probs)
        batch = [self.data[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        weights = (len(self.data) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, prio in zip(indices, priorities):
            self.priorities[int(idx)] = float(prio)

    def __len__(self) -> int:
        return len(self.data)


def _activation_from_name(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        act = _activation_from_name(activation)
        for _ in range(max(1, hidden_layers)):
            layers.append(nn.Linear(last_dim, hidden_units))
            layers.append(act)
            last_dim = hidden_units
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        act = _activation_from_name(activation)
        for _ in range(max(1, hidden_layers)):
            layers.append(nn.Linear(last_dim, hidden_units))
            layers.append(act)
            last_dim = hidden_units
        self.feature = nn.Sequential(*layers)
        self.value = nn.Linear(last_dim, 1)
        self.advantage = nn.Linear(last_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature(x)
        value = self.value(feats)
        adv = self.advantage(feats)
        adv = adv - adv.mean(dim=1, keepdim=True)
        return value + adv


class NoisyLinear(nn.Module):
    weight_epsilon: torch.Tensor
    bias_epsilon: torch.Tensor

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = float(sigma_init)
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str, sigma_init: float):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        act = _activation_from_name(activation)
        for _ in range(max(1, hidden_layers)):
            layers.append(NoisyLinear(last_dim, hidden_units, sigma_init))
            layers.append(act)
            last_dim = hidden_units
        layers.append(NoisyLinear(last_dim, output_dim, sigma_init))
        self.net = nn.Sequential(*layers)

    def reset_noise(self):
        for layer in self.net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistributionalQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, atoms: int, hidden_layers: int, hidden_units: int, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        act = _activation_from_name(activation)
        for _ in range(max(1, hidden_layers)):
            layers.append(nn.Linear(last_dim, hidden_units))
            layers.append(act)
            last_dim = hidden_units
        layers.append(nn.Linear(last_dim, output_dim * atoms))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim
        self.atoms = atoms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.view(-1, self.output_dim, self.atoms)


class RainbowQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, atoms: int, hidden_layers: int, hidden_units: int, activation: str, sigma_init: float):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        act = _activation_from_name(activation)
        for _ in range(max(1, hidden_layers)):
            layers.append(NoisyLinear(last_dim, hidden_units, sigma_init))
            layers.append(act)
            last_dim = hidden_units
        self.feature = nn.Sequential(*layers)
        self.value = NoisyLinear(last_dim, atoms, sigma_init)
        self.advantage = NoisyLinear(last_dim, output_dim * atoms, sigma_init)
        self.output_dim = output_dim
        self.atoms = atoms

    def reset_noise(self):
        for layer in self.feature:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        self.value.reset_noise()
        self.advantage.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature(x)
        value = self.value(feats).view(-1, 1, self.atoms)
        adv = self.advantage(feats).view(-1, self.output_dim, self.atoms)
        adv = adv - adv.mean(dim=1, keepdim=True)
        return value + adv


class BasePolicy:
    name: str = "Base"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        hidden_layers: int,
        hidden_units: int,
        batch_size: int,
        replay_size: int,
        activation: str,
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.schedule = EpsilonSchedule(eps_start, eps_end, eps_decay)
        self.epsilon = float(eps_start)
        self.batch_size = int(batch_size)
        self.replay = ReplayBuffer(replay_size)
        self.device = torch.device("cpu")
        self.warmup_steps = int(warmup_steps)
        self.grad_clip = float(grad_clip)

        self.q_net = QNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()

    def start_episode(self, episode_idx: int):
        self.epsilon = self.schedule.epsilon_for(episode_idx)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q_net(state_t)
            return int(torch.argmax(qvals, dim=1).item())

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.replay.add(state, action, reward, next_state, done)

    def can_learn(self) -> bool:
        return len(self.replay) >= max(self.batch_size, self.warmup_steps)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            max_next = self.q_net(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + (1.0 - dones_t) * self.gamma * max_next

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

    def clip_gradients(self):
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)


class DQNPolicy(BasePolicy):
    name = "DQN"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        hidden_layers: int,
        hidden_units: int,
        batch_size: int,
        replay_size: int,
        activation: str,
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
        target_update: int = 100,
    ):
        super().__init__(
            state_dim,
            action_dim,
            alpha,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            hidden_layers,
            hidden_units,
            batch_size,
            replay_size,
            activation,
            warmup_steps,
            grad_clip,
        )
        self.target_net = QNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.target_update = int(target_update)
        self._learn_steps = 0

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            max_next = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + (1.0 - dones_t) * self.gamma * max_next

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class DDQNPolicy(BasePolicy):
    name = "DDQN"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        hidden_layers: int,
        hidden_units: int,
        batch_size: int,
        replay_size: int,
        activation: str,
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
        target_update: int = 100,
    ):
        super().__init__(
            state_dim,
            action_dim,
            alpha,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            hidden_layers,
            hidden_units,
            batch_size,
            replay_size,
            activation,
            warmup_steps,
            grad_clip,
        )
        self.target_net = QNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.target_update = int(target_update)
        self._learn_steps = 0

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states_t), dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class PrioritizedDDQNPolicy(BasePolicy):
    name = "Prioritized DDQN"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        hidden_layers: int,
        hidden_units: int,
        batch_size: int,
        replay_size: int,
        activation: str,
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
        target_update: int = 100,
        prior_alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100000,
    ):
        super().__init__(
            state_dim,
            action_dim,
            alpha,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            hidden_layers,
            hidden_units,
            batch_size,
            replay_size,
            activation,
            warmup_steps,
            grad_clip,
        )
        self.replay = PrioritizedReplayBuffer(replay_size, alpha=prior_alpha)
        self.target_net = QNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.target_update = int(target_update)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_steps = int(beta_steps)
        self._learn_steps = 0

    def _beta_for_step(self) -> float:
        if self.beta_steps <= 0:
            return self.beta_end
        frac = min(1.0, self._learn_steps / self.beta_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        beta = self._beta_for_step()
        states, actions, rewards, next_states, dones, indices, weights = self.replay.sample(self.batch_size, beta=beta)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states_t), dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = torch.nn.functional.smooth_l1_loss(q_values, target, reduction="none")
        loss = (loss * weights_t).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

        td_errors = (target - q_values).detach().abs().squeeze(1).cpu().numpy() + 1e-6
        self.replay.update_priorities(indices, td_errors)

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class DuelingDDQNPolicy(BasePolicy):
    name = "Dueling DDQN"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        hidden_layers: int,
        hidden_units: int,
        batch_size: int,
        replay_size: int,
        activation: str,
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
        target_update: int = 100,
    ):
        super().__init__(
            state_dim,
            action_dim,
            alpha,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            hidden_layers,
            hidden_units,
            batch_size,
            replay_size,
            activation,
            warmup_steps,
            grad_clip,
        )
        self.q_net = DuelingQNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.target_update = int(target_update)
        self._learn_steps = 0

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states_t), dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class NoisyDQNPolicy(BasePolicy):
    name = "Noisy DQN"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        hidden_layers: int,
        hidden_units: int,
        batch_size: int,
        replay_size: int,
        activation: str,
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
        noisy_sigma: float = 0.5,
        target_update: int = 100,
    ):
        super().__init__(
            state_dim,
            action_dim,
            alpha,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            hidden_layers,
            hidden_units,
            batch_size,
            replay_size,
            activation,
            warmup_steps,
            grad_clip,
        )
        self.q_net = NoisyQNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation, noisy_sigma).to(self.device)
        self.target_net = NoisyQNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation, noisy_sigma).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.target_update = int(target_update)
        self._learn_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        # Noisy networks handle exploration internally
        self.q_net.reset_noise()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q_net(state_t)
            return int(torch.argmax(qvals, dim=1).item())

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.q_net.reset_noise()
        self.target_net.reset_noise()
        q_values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states_t), dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class DistributionalDQNPolicy(BasePolicy):
    name = "Distributional DQN"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        hidden_layers: int,
        hidden_units: int,
        batch_size: int,
        replay_size: int,
        activation: str,
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
        atoms: int = 51,
        v_min: float = -300.0,
        v_max: float = 300.0,
        target_update: int = 100,
    ):
        super().__init__(
            state_dim,
            action_dim,
            alpha,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            hidden_layers,
            hidden_units,
            batch_size,
            replay_size,
            activation,
            warmup_steps,
            grad_clip,
        )
        self.atoms = int(atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.target_update = int(target_update)
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms, device=self.device)

        self.q_net = DistributionalQNetwork(state_dim, action_dim, self.atoms, hidden_layers, hidden_units, activation).to(self.device)
        self.target_net = DistributionalQNetwork(state_dim, action_dim, self.atoms, hidden_layers, hidden_units, activation).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self._learn_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.q_net(state_t)
            probs = torch.softmax(logits, dim=2)
            q_vals = torch.sum(probs * self.support.view(1, 1, -1), dim=2)
            return int(torch.argmax(q_vals, dim=1).item())

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        logits = self.q_net(states_t)
        log_probs = torch.log_softmax(logits, dim=2)
        action_log_probs = log_probs[torch.arange(self.batch_size, device=self.device), actions_t]

        with torch.no_grad():
            next_logits = self.target_net(next_states_t)
            next_probs = torch.softmax(next_logits, dim=2)
            next_q = torch.sum(next_probs * self.support.view(1, 1, -1), dim=2)
            next_actions = torch.argmax(next_q, dim=1)
            next_dist = next_probs[torch.arange(self.batch_size, device=self.device), next_actions]

            t_z = rewards_t + (1.0 - dones_t) * self.gamma * self.support.view(1, -1)
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros(self.batch_size, self.atoms, device=self.device)
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size, device=self.device).long().unsqueeze(1)
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        loss = -(m * action_log_probs).sum(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class RainbowPolicy(BasePolicy):
    name = "Rainbow"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        hidden_layers: int,
        hidden_units: int,
        batch_size: int,
        replay_size: int,
        activation: str,
        warmup_steps: int = 1000,
        grad_clip: float = 10.0,
        atoms: int = 51,
        v_min: float = -300.0,
        v_max: float = 300.0,
        target_update: int = 100,
        noisy_sigma: float = 0.5,
        prior_alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100000,
    ):
        super().__init__(
            state_dim,
            action_dim,
            alpha,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            hidden_layers,
            hidden_units,
            batch_size,
            replay_size,
            activation,
            warmup_steps,
            grad_clip,
        )
        self.atoms = int(atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.target_update = int(target_update)
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms, device=self.device)
        self.replay = PrioritizedReplayBuffer(replay_size, alpha=prior_alpha)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_steps = int(beta_steps)

        self.q_net = RainbowQNetwork(state_dim, action_dim, self.atoms, hidden_layers, hidden_units, activation, noisy_sigma).to(self.device)
        self.target_net = RainbowQNetwork(state_dim, action_dim, self.atoms, hidden_layers, hidden_units, activation, noisy_sigma).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self._learn_steps = 0

    def _beta_for_step(self) -> float:
        if self.beta_steps <= 0:
            return self.beta_end
        frac = min(1.0, self._learn_steps / self.beta_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def select_action(self, state: np.ndarray) -> int:
        # Noisy networks handle exploration
        self.q_net.reset_noise()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.q_net(state_t)
            probs = torch.softmax(logits, dim=2)
            q_vals = torch.sum(probs * self.support.view(1, 1, -1), dim=2)
            return int(torch.argmax(q_vals, dim=1).item())

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        beta = self._beta_for_step()
        states, actions, rewards, next_states, dones, indices, weights = self.replay.sample(self.batch_size, beta=beta)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.q_net.reset_noise()
        logits = self.q_net(states_t)
        log_probs = torch.log_softmax(logits, dim=2)
        action_log_probs = log_probs[torch.arange(self.batch_size, device=self.device), actions_t]

        with torch.no_grad():
            next_logits = self.q_net(next_states_t)
            next_probs = torch.softmax(next_logits, dim=2)
            next_q = torch.sum(next_probs * self.support.view(1, 1, -1), dim=2)
            next_actions = torch.argmax(next_q, dim=1)

            self.target_net.reset_noise()
            target_logits = self.target_net(next_states_t)
            target_probs = torch.softmax(target_logits, dim=2)
            next_dist = target_probs[torch.arange(self.batch_size, device=self.device), next_actions]

            t_z = rewards_t + (1.0 - dones_t) * self.gamma * self.support.view(1, -1)
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros(self.batch_size, self.atoms, device=self.device)
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size, device=self.device).long().unsqueeze(1)
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        per_sample_loss = -(m * action_log_probs).sum(dim=1, keepdim=True)
        loss = (per_sample_loss * weights_t).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

        td_errors = per_sample_loss.detach().squeeze(1).cpu().numpy() + 1e-6
        self.replay.update_priorities(indices, td_errors)

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class Agent:
    def __init__(self, env: LunarLanderEnv, policy: BasePolicy):
        self.env = env
        self.policy = policy

    def run_episode(
        self,
        episode_idx: int,
        max_steps: int,
        step_delay: float,
        render_callback: Optional[Callable[[object], None]] = None,
        stop_event: Optional[Event] = None,
    ) -> float:
        self.policy.start_episode(episode_idx)
        state = self.env.reset()
        total = 0.0

        for _ in range(max_steps):
            if stop_event is not None and stop_event.is_set():
                break
            action_idx = self.policy.select_action(state)
            action = self.env.action_from_index(action_idx)
            next_state, reward, done, _ = self.env.step(action)
            self.policy.store(state, action_idx, reward, next_state, done)
            if self.policy.can_learn():
                self.policy.learn()
            total += reward
            if render_callback is not None:
                render_callback(self.env.render())
            if step_delay > 0:
                time.sleep(step_delay)
            state = next_state
            if done:
                break
        return total
