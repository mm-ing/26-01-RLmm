import tkinter as tk

from taxi_gui import TaxiGUI
from taxi_logic import (
    Agent,
    TaxiEnv,
    QLearningPolicy,
    SarsaPolicy,
    ExpectedSarsaPolicy,
)


def main():
    env = TaxiEnv(is_raining=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policies = [
        QLearningPolicy(n_states, n_actions, alpha=0.2, gamma=0.8, eps_start=1.0, eps_end=0.05, total_episodes=1000),
        SarsaPolicy(n_states, n_actions, alpha=0.2, gamma=0.8, eps_start=1.0, eps_end=0.05, total_episodes=1000),
        ExpectedSarsaPolicy(n_states, n_actions, alpha=0.2, gamma=0.8, eps_start=1.0, eps_end=0.05, total_episodes=1000),
    ]
    agent = Agent(env, policies[0])
    root = tk.Tk()
    app = TaxiGUI(root, env, agent, policies)
    root.mainloop()


if __name__ == "__main__":
    main()
