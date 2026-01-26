import tkinter as tk

from frozen_lake_gui import FrozenLakeGUI
from frozen_lake_logic import (
    Agent,
    FrozenLakeEnv,
    QLearningPolicy,
    SarsaPolicy,
    ExpectedSarsaPolicy,
)


def main():
    env = FrozenLakeEnv(map_name="4x4", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policies = [
        QLearningPolicy(n_states, n_actions, alpha=0.2, gamma=0.8, eps_start=1.0, eps_end=0.05, total_episodes=1000),
        SarsaPolicy(n_states, n_actions, alpha=0.2, gamma=0.8, eps_start=1.0, eps_end=0.05, total_episodes=1000),
        ExpectedSarsaPolicy(n_states, n_actions, alpha=0.2, gamma=0.8, eps_start=1.0, eps_end=0.05, total_episodes=1000),
    ]
    agent = Agent(env, policies[0])
    root = tk.Tk()
    app = FrozenLakeGUI(root, env, agent, policies)
    root.mainloop()


if __name__ == "__main__":
    main()
