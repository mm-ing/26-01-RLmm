"""
HalfCheetah RL Application
Entry point for HalfCheetah reinforcement learning training
"""

import tkinter as tk
from half_cheetah_logic import (
    HalfCheetahEnvironment,
    PPO,
    TD3,
    SAC,
    Agent
)
from half_cheetah_gui import HalfCheetahGUI


def main():
    """Main entry point"""
    # Create environment (start without rendering, can be enabled via GUI)
    environment = HalfCheetahEnvironment(
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        render_mode=None
    )
    
    # Get dimensions
    state_dim = environment.get_state_dim()
    action_dim = environment.get_action_dim()
    
    # Create policy classes dictionary
    policies = {
        'PPO': PPO,
        'TD3': TD3,
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
    gui = HalfCheetahGUI(root, environment, agent, policies)
    
    # Run GUI
    gui.run()
    
    # Cleanup
    environment.close()


if __name__ == "__main__":
    main()
