"""
BipedalWalker RL Application
Entry point for BipedalWalker reinforcement learning training
"""

import tkinter as tk
from bipedal_walker_logic import (
    BipedalWalkerEnvironment,
    RainbowDQN,
    A2C,
    TRPO,
    PPO,
    SAC,
    Agent
)
from bipedal_walker_gui import BipedalWalkerGUI


def main():
    """Main entry point"""
    # Create environment (start without rendering, can be enabled via GUI)
    environment = BipedalWalkerEnvironment(hardcore=False, render_mode=None)
    
    # Get dimensions
    state_dim = environment.get_state_dim()
    action_dim = environment.get_action_dim()
    
    # Create policy classes dictionary
    policies = {
        'Rainbow': RainbowDQN,
        'A2C': A2C,
        'TRPO': TRPO,
        'PPO': PPO,
        'SAC': SAC
    }
    
    # Create default policy instance
    default_policy = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        lr=3e-4,
        gamma=0.99
    )
    
    # Create agent
    agent = Agent(environment, default_policy)
    
    # Create GUI
    root = tk.Tk()
    gui = BipedalWalkerGUI(root, environment, agent, policies)
    
    # Run GUI
    gui.run()
    
    # Cleanup
    environment.close()


if __name__ == "__main__":
    main()
