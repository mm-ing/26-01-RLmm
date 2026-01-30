from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CartPoleEnv:
    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self) -> np.ndarray:
        obs, _info = self.env.reset()
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
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

        self.q_net = QNetwork(state_dim, action_dim, hidden_layers, hidden_units, activation).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

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

    def learn(self):
        raise NotImplementedError


class DQNPolicy(BasePolicy):
    name = "DQN"

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
        self.optimizer.step()


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
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class Agent:
    def __init__(self, env: CartPoleEnv, policy: BasePolicy):
        self.env = env
        self.policy = policy

    def run_episode(
        self,
        episode_idx: int,
        max_steps: int,
        step_delay: float,
        render_callback: Optional[Callable[[object], None]] = None,
        stop_event: Optional[object] = None,
    ) -> float:
        self.policy.start_episode(episode_idx)
        state = self.env.reset()
        total = 0.0

        for _ in range(max_steps):
            if stop_event is not None and stop_event.is_set():
                break
            action = self.policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.policy.store(state, action, reward, next_state, done)
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
