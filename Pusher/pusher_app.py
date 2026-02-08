"""
Pusher RL Application
Entry point for Pusher reinforcement learning training
"""

import tkinter as tk
from pusher_logic import PusherEnvironment, DDPG, TD3, SAC, Agent
from pusher_gui import PusherGUI


def main():
    """Main entry point"""
    environment = PusherEnvironment(
        reward_near_weight=0.5,
        reward_dist_weight=1.0,
        reward_control_weight=0.1,
        render_mode=None
    )

    state_dim = environment.get_state_dim()
    action_dim = environment.get_action_dim()

    policies = {
        "DDPG": DDPG,
        "TD3": TD3,
        "SAC": SAC
    }

    default_policy = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        lr=3e-4,
        gamma=0.99
    )

    agent = Agent(environment, default_policy)

    root = tk.Tk()
    gui = PusherGUI(root, environment, agent, policies)
    gui.run()

    environment.close()


if __name__ == "__main__":
    main()
