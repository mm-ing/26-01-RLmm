from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np


class CliffWalkingEnv:
    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        self.env = gym.make("CliffWalking-v1", render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        obs, _info = self.env.reset()
        return obs

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, float(reward), done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


@dataclass
class EpsilonSchedule:
    eps_start: float
    eps_end: float
    total_episodes: int

    def epsilon_for(self, episode_idx: int) -> float:
        if self.total_episodes <= 1:
            return self.eps_end
        span = max(1, self.total_episodes - 1)
        frac = min(1.0, max(0.0, episode_idx / span))
        return max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * frac)


class BasePolicy:
    name: str = "Base"

    def __init__(self, n_states: int, n_actions: int, alpha: float, gamma: float, eps_start: float, eps_end: float, total_episodes: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.schedule = EpsilonSchedule(eps_start, eps_end, total_episodes)
        self.epsilon = eps_start
        self.q = np.zeros((n_states, n_actions), dtype=np.float32)

    def start_episode(self, episode_idx: int):
        self.epsilon = self.schedule.epsilon_for(episode_idx)

    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        qvals = self.q[state]
        best = np.flatnonzero(qvals == qvals.max())
        return int(random.choice(best))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool, **kwargs):
        raise NotImplementedError

    def value_table(self):
        return np.max(self.q, axis=1)


class QLearningPolicy(BasePolicy):
    name = "Q-learning"

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool, **kwargs):
        best_next = float(np.max(self.q[next_state]))
        target = reward if done else reward + self.gamma * best_next
        td = target - self.q[state, action]
        self.q[state, action] += self.alpha * td


class SarsaPolicy(BasePolicy):
    name = "SARSA"

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool, **kwargs):
        next_action = kwargs.get("next_action", 0)
        next_q = 0.0 if done else float(self.q[next_state, next_action])
        target = reward + self.gamma * next_q
        td = target - self.q[state, action]
        self.q[state, action] += self.alpha * td


class ExpectedSarsaPolicy(BasePolicy):
    name = "Expected SARSA"

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool, **kwargs):
        if done:
            expected_next = 0.0
        else:
            qvals = self.q[next_state]
            max_q = float(np.max(qvals))
            best_actions = np.flatnonzero(qvals == max_q)
            n = self.n_actions
            probs = np.full(n, self.epsilon / n, dtype=np.float32)
            probs[best_actions] += (1.0 - self.epsilon) / max(1, len(best_actions))
            expected_next = float(np.sum(probs * qvals))
        target = reward + self.gamma * expected_next
        td = target - self.q[state, action]
        self.q[state, action] += self.alpha * td


class Agent:
    def __init__(self, env: CliffWalkingEnv, policy: BasePolicy):
        self.env = env
        self.policy = policy

    def run_episode(
        self,
        episode_idx: int,
        total_episodes: int,
        max_steps: int,
        step_delay: float,
        render_callback: Optional[Callable[[object], None]] = None,
        stop_event: Optional[object] = None,
    ) -> float:
        self.policy.schedule.total_episodes = total_episodes
        self.policy.start_episode(episode_idx)
        state = int(self.env.reset())
        total = 0.0
        steps = 0
        if isinstance(self.policy, SarsaPolicy):
            action = self.policy.select_action(state)
            while steps < max_steps:
                if stop_event is not None and stop_event.is_set():
                    break
                next_state, reward, done, _ = self.env.step(action)
                next_state = int(next_state)
                next_action = self.policy.select_action(next_state) if not done else 0
                self.policy.update(state, action, reward, next_state, done, next_action=next_action)
                total += reward
                steps += 1
                if render_callback is not None:
                    render_callback(self.env.render())
                if step_delay > 0:
                    time.sleep(step_delay)
                state, action = next_state, next_action
                if done:
                    break
        else:
            while steps < max_steps:
                if stop_event is not None and stop_event.is_set():
                    break
                action = self.policy.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = int(next_state)
                self.policy.update(state, action, reward, next_state, done)
                total += reward
                steps += 1
                if render_callback is not None:
                    render_callback(self.env.render())
                if step_delay > 0:
                    time.sleep(step_delay)
                state = next_state
                if done:
                    break
        return total
