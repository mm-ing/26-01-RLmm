"""Walker2D Application - Entry point for RL training"""
import tkinter as tk
from walker_2d_logic import Walker2DEnvironment, PPO, TD3, SAC, Agent
from walker_2d_gui import Walker2DGUI


def main():
    """Main entry point"""
    # Create environment
    env = Walker2DEnvironment(render_mode=None)
    
    # Create policies
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    policies = {
        'PPO': PPO(state_dim, action_dim),
        'TD3': TD3(state_dim, action_dim),
        'SAC': SAC(state_dim, action_dim)
    }
    
    # Create agent with default policy
    default_policy = policies['PPO']
    agent = Agent(env, default_policy)
    
    # Create GUI
    root = tk.Tk()
    gui = Walker2DGUI(root, env, agent, policies)
    
    # Run
    gui.run()


if __name__ == '__main__':
    main()
