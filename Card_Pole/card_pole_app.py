import tkinter as tk

from card_pole_gui import CardPoleGUI
from card_pole_logic import (
    Agent,
    CartPoleEnv,
    DQNPolicy,
    DDQNPolicy,
)


def main():
    env = CartPoleEnv()
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    policies = [
        DQNPolicy(state_dim, action_dim, alpha=0.2, gamma=0.8, eps_start=1.0, eps_end=0.05, eps_decay=0.05, hidden_layers=2, hidden_units=64, batch_size=64, replay_size=2000, activation="ReLU"),
        DDQNPolicy(state_dim, action_dim, alpha=0.2, gamma=0.8, eps_start=1.0, eps_end=0.05, eps_decay=0.05, hidden_layers=2, hidden_units=64, batch_size=64, replay_size=2000, activation="ReLU"),
    ]
    agent = Agent(env, policies[0])
    root = tk.Tk()
    app = CardPoleGUI(root, env, agent, policies)
    root.mainloop()


if __name__ == "__main__":
    main()
