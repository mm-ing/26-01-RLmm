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


class PendulumEnv:
    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None, action_bins: int = 21):
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)
        self.action_bins = max(2, int(action_bins))
        if not isinstance(self.env.action_space, spaces.Box):
            raise ValueError("Pendulum action space must be Box.")
        low = float(self.env.action_space.low[0])
        high = float(self.env.action_space.high[0])
        self.action_values = np.linspace(low, high, self.action_bins, dtype=np.float32)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def action_from_index(self, action_idx: int) -> np.ndarray:
        idx = int(np.clip(action_idx, 0, self.action_bins - 1))
        return np.asarray([self.action_values[idx]], dtype=np.float32)

    def reset(self) -> np.ndarray:
        obs, _info = self.env.reset()
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: np.ndarray):
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


class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = int(dim)
        self.eps = float(eps)
        self.count = 0
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.m2 = np.zeros(self.dim, dtype=np.float32)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float32)
        return self.m2 / (self.count - 1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)


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
        normalize_states: bool = True,
        reward_scale: float = 0.1,
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
        self.normalize_states = bool(normalize_states)
        self.reward_scale = float(reward_scale)
        self.grad_clip = float(grad_clip)
        self._normalizer = RunningNorm(state_dim) if self.normalize_states else None

        self.q_net = QNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()

    def start_episode(self, episode_idx: int):
        self.epsilon = self.schedule.epsilon_for(episode_idx)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            if self._normalizer is not None:
                state = self._normalizer.normalize(state)
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q_net(state_t)
            return int(torch.argmax(qvals, dim=1).item())

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        if self._normalizer is not None:
            self._normalizer.update(state)
            self._normalizer.update(next_state)
            state = self._normalizer.normalize(state)
            next_state = self._normalizer.normalize(next_state)
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
        normalize_states: bool = True,
        reward_scale: float = 0.1,
        grad_clip: float = 10.0,
        target_update: int = 50,
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
            normalize_states,
            reward_scale,
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
        normalize_states: bool = True,
        reward_scale: float = 0.1,
        grad_clip: float = 10.0,
        target_update: int = 10,
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
            normalize_states,
            reward_scale,
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
        normalize_states: bool = True,
        reward_scale: float = 0.1,
        grad_clip: float = 10.0,
        target_update: int = 10,
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
            normalize_states,
            reward_scale,
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
        normalize_states: bool = True,
        reward_scale: float = 0.1,
        grad_clip: float = 10.0,
        noisy_sigma: float = 0.5,
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
            normalize_states,
            reward_scale,
            grad_clip,
        )
        self.q_net = NoisyQNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation, noisy_sigma).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        self.q_net.reset_noise()
        with torch.no_grad():
            if self._normalizer is not None:
                state = self._normalizer.normalize(state)
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
        q_values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            max_next = self.q_net(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + (1.0 - dones_t) * self.gamma * max_next

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()


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
        normalize_states: bool = True,
        reward_scale: float = 0.1,
        grad_clip: float = 10.0,
        atoms: int = 51,
        v_min: float = -7.0,
        v_max: float = -5.0,
        target_update: int = 10,
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
            normalize_states,
            reward_scale,
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
            if self._normalizer is not None:
                state = self._normalizer.normalize(state)
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
            eq_mask = (u == l).float()
            if eq_mask.any():
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * eq_mask).view(-1))

        loss = -(m * action_log_probs).sum(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class Agent:
    def __init__(self, env: PendulumEnv, policy: BasePolicy):
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
            scaled_reward = reward * self.policy.reward_scale
            self.policy.store(state, action_idx, scaled_reward, next_state, done)
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
