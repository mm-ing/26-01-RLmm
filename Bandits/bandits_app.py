from bandit_gui import main
from bandit_logic import OpenArmedBandit, Agent
from bandit_gui import BanditGUI
import tkinter as tk


def run():
    # initialize three bandits and an agent, inject into GUI
    probs = [0.2, 0.5, 0.8]
    envs = [OpenArmedBandit(p) for p in probs]
    agent = Agent(n_arms=len(envs))
    root = tk.Tk()
    app = BanditGUI(root, envs=envs, agent=agent)
    root.mainloop()


if __name__ == "__main__":
    run()
