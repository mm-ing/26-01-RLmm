import random
from collections import deque
from typing import List, Dict, Tuple


class OpenArmedBandit:
    """Simple Bernoulli bandit with success probability `reward_prob`."""

    def __init__(self, reward_prob: float):
        self.reward_prob = float(reward_prob)

    def pull(self) -> int:
        """Return 1 for success, 0 for failure (Bernoulli)."""
        return 1 if random.random() < self.reward_prob else 0


class Agent:
    """Epsilon-greedy agent with multiplicative epsilon decay and optional finite memory.

    - `memory == 0` means use full history; otherwise keep a per-arm deque(maxlen=memory).
    - `decay` multiplies `epsilon` after each agent action: `epsilon *= decay`.
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1, decay: float = 1.0, memory: int = 0, policy=None, policy_name: str = "epsilon"):
        self.n_arms = int(n_arms)
        self.memory = int(memory)
        self.epsilon = float(epsilon)
        self.decay = float(decay)

        # Per-arm recent-history used to compute empirical means (deque per arm).
        if self.memory > 0:
            self._history: List[deque] = [deque(maxlen=self.memory) for _ in range(self.n_arms)]
        else:
            # deque without maxlen => full history
            self._history = [deque() for _ in range(self.n_arms)]

        # Cumulative stats (agent-only): totals across all time for the agent's actions
        self.pulls: List[int] = [0] * self.n_arms
        self.successes: List[int] = [0] * self.n_arms
        self.cumulative_reward: int = 0
        # Policy: either provided policy object or create one from name/params
        if policy is not None:
            self.policy = policy
        else:
            if policy_name.lower().startswith("thompson"):
                self.policy = ThompsonSamplingPolicy(self.n_arms)
            else:
                # default: epsilon-greedy policy uses provided epsilon/decay
                self.policy = EpsilonGreedyPolicy(epsilon, decay)
        self.policy_name = self.policy.__class__.__name__

    def _estimated_means(self) -> List[float]:
        means: List[float] = []
        for i in range(self.n_arms):
            hist = self._history[i]
            if len(hist) > 0:
                means.append(sum(hist) / len(hist))
            else:
                # fall back to cumulative frequency if available
                if self.pulls[i] > 0:
                    means.append(self.successes[i] / self.pulls[i])
                else:
                    means.append(0.0)
        return means

    def select_action(self) -> int:
        """Select an arm index using epsilon-greedy policy."""
        return self.policy.select_action(self)

    def update(self, arm: int, reward: int) -> None:
        """Record the outcome of pulling `arm` with observed `reward` (0 or 1)."""
        self._history[arm].append(int(bool(reward)))
        self.pulls[arm] += 1
        self.successes[arm] += int(bool(reward))
        self.cumulative_reward += int(bool(reward))
        # let the policy update any internal state (e.g. decay epsilon or update posteriors)
        if hasattr(self.policy, "update"):
            try:
                self.policy.update(arm, int(bool(reward)))
            except TypeError:
                # fallback for policies expecting different signature
                self.policy.update(arm, reward)

    def run(self, envs: List[OpenArmedBandit], n_loops: int) -> Dict:
        """Run `n_loops` actions in the provided environments.

        Returns a summary dict containing per-arm pulls/successes, total rewards and final epsilon.
        """
        rewards: List[int] = []
        for _ in range(int(n_loops)):
            arm = self.select_action()
            reward = envs[arm].pull()
            self.update(arm, reward)
            rewards.append(reward)

        return {
            "total_rewards": sum(rewards),
            "rewards": rewards,
            "pulls": list(self.pulls),
            "successes": list(self.successes),
            "epsilon": self.epsilon,
        }

    def run_one(self, envs: List[OpenArmedBandit]) -> Tuple[int, int]:
        """Perform a single agent action: select, pull, update, decay epsilon.

        Returns (arm_index, reward).
        """
        arm = self.select_action()
        reward = envs[arm].pull()
        self.update(arm, reward)
        return arm, reward

    def stats(self) -> Dict:
        """Return simple agent stats."""
        rates: List[float] = [0.0] * self.n_arms
        for i in range(self.n_arms):
            pulls_i = int(self.pulls[i]) if i < len(self.pulls) and self.pulls[i] is not None else 0
            succ_i = int(self.successes[i]) if i < len(self.successes) and self.successes[i] is not None else 0
            rates[i] = (succ_i / pulls_i) if pulls_i > 0 else 0.0
        return {
            "pulls": list(self.pulls),
            "successes": list(self.successes),
            "rates": rates,
            "cumulative_reward": self.cumulative_reward,
            "policy": self.policy_name,
            "epsilon": getattr(self.policy, "epsilon", None),
        }


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy with multiplicative decay."""

    def __init__(self, epsilon: float = 0.1, decay: float = 1.0):
        self.epsilon = float(epsilon)
        self.decay = float(decay)

    def select_action(self, agent: Agent) -> int:
        if random.random() < self.epsilon:
            return random.randrange(agent.n_arms)
        means = agent._estimated_means()
        best = max(means)
        best_indices = [i for i, m in enumerate(means) if m == best]
        return random.choice(best_indices)

    def update(self, arm: int, reward: int) -> None:
        self.epsilon *= self.decay


class ThompsonSamplingPolicy:
    """Thompson Sampling for Bernoulli bandits using Beta priors."""

    def __init__(self, n_arms: int):
        self.n_arms = int(n_arms)
        # start with uniform Beta(1,1) prior
        self.alphas: List[float] = [1.0] * self.n_arms
        self.betas: List[float] = [1.0] * self.n_arms

    def select_action(self, agent: Agent) -> int:
        samples = [random.betavariate(self.alphas[i], self.betas[i]) for i in range(self.n_arms)]
        best = max(samples)
        best_indices = [i for i, s in enumerate(samples) if s == best]
        return random.choice(best_indices)

    def update(self, arm: int, reward: int) -> None:
        if int(bool(reward)):
            self.alphas[arm] += 1.0
        else:
            self.betas[arm] += 1.0
