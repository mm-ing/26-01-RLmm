import tkinter as tk

from grid_gui import GridGUI
from grid_logic import (
    Agent,
    GridWorld,
    MonteCarloPolicies,
    SarsaPolicies,
    ExpSarsaPolicies,
    QlearningPolicies,
)


def main():
    env = GridWorld()
    policies = [
        MonteCarloPolicies(gamma=0.9),
        SarsaPolicies(),
        ExpSarsaPolicies(),
        QlearningPolicies(),
    ]
    agent = Agent(env, policies[0])
    root = tk.Tk()
    app = GridGUI(root, env, agent, policies)
    root.mainloop()


if __name__ == "__main__":
    main()
