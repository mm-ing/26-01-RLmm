from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

Action = int
State = Tuple[int, int]


class GridWorld:
    """Deterministic GridWorld.

    Coordinates: (x, y), origin (0,0) top-left.
    Actions: 0=Up,1=Down,2=Left,3=Right
    Reward: -1 per step, 0 at goal. Episode ends at goal or max_steps.
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 3,
        start: State = (0, 2),
        goal: State = (4, 2),
        blocked: Optional[List[State]] = None,
        max_steps: int = 20,
        seed: Optional[int] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.blocked = set(blocked or [])
        self.max_steps = int(max_steps)
        self._rng = random.Random(seed)

        self.state: State = self.start
        self.steps = 0
        self.reset()

    def reset(self) -> State:
        self.state = tuple(self.start)
        self.steps = 0
        return self.state

    def in_bounds(self, pos: State) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, pos: State) -> bool:
        return pos in self.blocked

    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        x, y = self.state
        if action == 0:
            nxt = (x, y - 1)
        elif action == 1:
            nxt = (x, y + 1)
        elif action == 2:
            nxt = (x - 1, y)
        elif action == 3:
            nxt = (x + 1, y)
        else:
            nxt = (x, y)

        if not self.in_bounds(nxt) or self.is_blocked(nxt):
            nxt = (x, y)

        self.state = nxt
        self.steps += 1
        done = (nxt == self.goal) or (self.steps >= self.max_steps)
        reward = 0.0 if nxt == self.goal else -1.0
        return nxt, reward, done, {}

    def available_actions(self) -> List[Action]:
        return [0, 1, 2, 3]

    def to_matrix(self) -> List[List[int]]:
        """Return a simple matrix with codes: 0=free,1=blocked,2=start,3=goal,4=agent"""
        mat = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for bx, by in self.blocked:
            if 0 <= bx < self.width and 0 <= by < self.height:
                mat[by][bx] = 1
        sx, sy = self.start
        gx, gy = self.goal
        if 0 <= sx < self.width and 0 <= sy < self.height:
            mat[sy][sx] = 2
        if 0 <= gx < self.width and 0 <= gy < self.height:
            mat[gy][gx] = 3
        ax, ay = self.state
        if 0 <= ax < self.width and 0 <= ay < self.height:
            mat[ay][ax] = 4
        return mat


class BasePolicy:
    name: str = "Base"

    def start_episode(self, episode_idx: int):
        pass

    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        raise NotImplementedError

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool, **kwargs):
        pass

    def end_episode(self, trajectory: List[Dict]):
        pass

    def value_table(self) -> Dict[State, float]:
        return {}

    def q_table(self) -> Dict[State, List[float]]:
        return {}


class EpsilonGreedyMixin:
    def _epsilon_greedy(self, q: List[float], available_actions: List[Action], epsilon: float) -> Action:
        if random.random() < epsilon:
            return random.choice(available_actions)
        best = max(available_actions, key=lambda a: q[a])
        return best

    def _expected_value(self, q: List[float], available_actions: List[Action], epsilon: float) -> float:
        max_q = max(q[a] for a in available_actions)
        best_actions = [a for a in available_actions if q[a] == max_q]
        n = len(available_actions)
        exp_val = 0.0
        for a in available_actions:
            if a in best_actions:
                prob = (1 - epsilon) / len(best_actions) + epsilon / n
            else:
                prob = epsilon / n
            exp_val += prob * q[a]
        return exp_val


class QlearningPolicies(BasePolicy, EpsilonGreedyMixin):
    name = "Q-learning"

    def __init__(self, n_actions: int = 4, alpha: float = 0.1, gamma: float = 0.9, epsilon_max: float = 1.0, epsilon_min: float = 0.05):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.q: Dict[State, List[float]] = {}

    def start_episode(self, episode_idx: int):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * math.exp(-0.005 * episode_idx))

    def _ensure(self, state: State):
        if state not in self.q:
            self.q[state] = [0.0] * self.n_actions

    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._ensure(state)
        return self._epsilon_greedy(self.q[state], available_actions, self.epsilon)

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool, **kwargs):
        self._ensure(state)
        self._ensure(next_state)
        best_next = max(self.q[next_state])
        td = reward + (0.0 if done else self.gamma * best_next) - self.q[state][action]
        self.q[state][action] += self.alpha * td

    def value_table(self) -> Dict[State, float]:
        return {s: max(q) for s, q in self.q.items()}

    def q_table(self) -> Dict[State, List[float]]:
        return self.q


class SarsaPolicies(BasePolicy, EpsilonGreedyMixin):
    name = "SARSA"

    def __init__(self, n_actions: int = 4, alpha: float = 0.1, gamma: float = 0.9, epsilon_max: float = 1.0, epsilon_min: float = 0.05):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.q: Dict[State, List[float]] = {}

    def start_episode(self, episode_idx: int):
        self.epsilon = max(self.epsilon_min, 1 / (1 + 0.001 * episode_idx))

    def _ensure(self, state: State):
        if state not in self.q:
            self.q[state] = [0.0] * self.n_actions

    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._ensure(state)
        return self._epsilon_greedy(self.q[state], available_actions, self.epsilon)

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool, **kwargs):
        next_action = kwargs.get("next_action")
        self._ensure(state)
        self._ensure(next_state)
        next_q = 0.0 if done else self.q[next_state][next_action]
        td = reward + self.gamma * next_q - self.q[state][action]
        self.q[state][action] += self.alpha * td

    def value_table(self) -> Dict[State, float]:
        return {s: max(q) for s, q in self.q.items()}

    def q_table(self) -> Dict[State, List[float]]:
        return self.q


class ExpSarsaPolicies(BasePolicy, EpsilonGreedyMixin):
    name = "Expected SARSA"

    def __init__(self, n_actions: int = 4, alpha: float = 0.1, gamma: float = 0.9, epsilon_max: float = 1.0, epsilon_min: float = 0.05):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.q: Dict[State, List[float]] = {}

    def start_episode(self, episode_idx: int):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * math.exp(-0.0005 * episode_idx))

    def _ensure(self, state: State):
        if state not in self.q:
            self.q[state] = [0.0] * self.n_actions

    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._ensure(state)
        return self._epsilon_greedy(self.q[state], available_actions, self.epsilon)

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool, **kwargs):
        self._ensure(state)
        self._ensure(next_state)
        expected_next = 0.0 if done else self._expected_value(self.q[next_state], list(range(self.n_actions)), self.epsilon)
        td = reward + self.gamma * expected_next - self.q[state][action]
        self.q[state][action] += self.alpha * td

    def value_table(self) -> Dict[State, float]:
        return {s: max(q) for s, q in self.q.items()}

    def q_table(self) -> Dict[State, List[float]]:
        return self.q


class MonteCarloPolicies(BasePolicy, EpsilonGreedyMixin):
    name = "Monte Carlo"

    def __init__(self, n_actions: int = 4, gamma: float = 0.9, epsilon_min: float = 0.05):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon = 1.0
        self.returns_sum: Dict[Tuple[State, Action], float] = {}
        self.returns_count: Dict[Tuple[State, Action], int] = {}
        self.q: Dict[State, List[float]] = {}

    def start_episode(self, episode_idx: int):
        self.epsilon = max(self.epsilon_min, 1 / (1 + 0.001 * episode_idx))

    def _ensure(self, state: State):
        if state not in self.q:
            self.q[state] = [0.0] * self.n_actions

    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._ensure(state)
        return self._epsilon_greedy(self.q[state], available_actions, self.epsilon)

    def end_episode(self, trajectory: List[Dict]):
        # every-visit MC on-policy with discounted returns
        G = 0.0
        for t in reversed(range(len(trajectory))):
            step = trajectory[t]
            state, action, reward = step["state"], step["action"], step["reward"]
            G = reward + self.gamma * G
            key = (state, action)
            self.returns_sum[key] = self.returns_sum.get(key, 0.0) + G
            self.returns_count[key] = self.returns_count.get(key, 0) + 1
            self._ensure(state)
            self.q[state][action] = self.returns_sum[key] / self.returns_count[key]

    def value_table(self) -> Dict[State, float]:
        return {s: max(q) for s, q in self.q.items()}

    def q_table(self) -> Dict[State, List[float]]:
        return self.q


class Agent:
    def __init__(self, env: GridWorld, policy: BasePolicy):
        self.env = env
        self.policy = policy
        self.episode_idx = 0
        self.last_trajectory: List[Dict] = []

    def set_policy(self, policy: BasePolicy):
        self.policy = policy

    def start_episode(self):
        self.policy.start_episode(self.episode_idx)
        state = self.env.reset()
        self.last_trajectory = []
        return state

    def run_episode(self, max_steps: Optional[int] = None) -> Tuple[float, int]:
        max_steps = max_steps if max_steps is not None else self.env.max_steps
        state = self.start_episode()
        total = 0.0
        steps = 0
        available = self.env.available_actions()

        if isinstance(self.policy, SarsaPolicies):
            action = self.policy.select_action(state, available)
            for _ in range(int(max_steps)):
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.policy.select_action(next_state, available) if not done else 0
                self.policy.update(state, action, reward, next_state, done, next_action=next_action)
                self.last_trajectory.append({
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                })
                total += reward
                steps += 1
                state, action = next_state, next_action
                if done:
                    break
        else:
            for _ in range(int(max_steps)):
                action = self.policy.select_action(state, available)
                next_state, reward, done, _ = self.env.step(action)
                self.policy.update(state, action, reward, next_state, done)
                self.last_trajectory.append({
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                })
                total += reward
                steps += 1
                state = next_state
                if done:
                    break

        self.policy.end_episode(self.last_trajectory)
        self.episode_idx += 1
        return total, steps


# Pathfinding helpers

def bfs_path(env: GridWorld, start: State, goal: State) -> List[State]:
    from collections import deque
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for a in env.available_actions():
            nxt, _, _, _ = _simulate_step(env, cur, a)
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if goal not in prev:
        return [start]
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def a_star_path(env: GridWorld, start: State, goal: State) -> List[State]:
    import heapq

    def h(a: State, b: State):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    pq = [(0 + h(start, goal), 0, start, None)]
    prev = {start: None}
    cost = {start: 0}
    while pq:
        _, g, cur, _ = heapq.heappop(pq)
        if cur == goal:
            break
        for a in env.available_actions():
            nxt, _, _, _ = _simulate_step(env, cur, a)
            ng = g + 1
            if nxt not in cost or ng < cost[nxt]:
                cost[nxt] = ng
                prev[nxt] = cur
                heapq.heappush(pq, (ng + h(nxt, goal), ng, nxt, cur))
    if goal not in prev:
        return [start]
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def random_path(env: GridWorld, start: State, goal: State, max_steps: int) -> List[State]:
    path = [start]
    cur = start
    for _ in range(int(max_steps)):
        a = random.choice(env.available_actions())
        nxt, _, _, _ = _simulate_step(env, cur, a)
        path.append(nxt)
        cur = nxt
        if cur == goal:
            break
    return path


def _simulate_step(env: GridWorld, state: State, action: Action) -> Tuple[State, float, bool, Dict]:
    x, y = state
    if action == 0:
        nxt = (x, y - 1)
    elif action == 1:
        nxt = (x, y + 1)
    elif action == 2:
        nxt = (x - 1, y)
    elif action == 3:
        nxt = (x + 1, y)
    else:
        nxt = (x, y)
    if not env.in_bounds(nxt) or env.is_blocked(nxt):
        nxt = (x, y)
    return nxt, 0.0, False, {}


def path_to_actions(path: List[State]) -> List[Action]:
    actions: List[Action] = []
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        if x1 == x0 and y1 == y0 - 1:
            actions.append(0)
        elif x1 == x0 and y1 == y0 + 1:
            actions.append(1)
        elif x1 == x0 - 1 and y1 == y0:
            actions.append(2)
        elif x1 == x0 + 1 and y1 == y0:
            actions.append(3)
        else:
            actions.append(0)
    return actions
