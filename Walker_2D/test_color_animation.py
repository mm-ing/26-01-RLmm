"""Test colored plots and animation fix"""
import tkinter as tk
from walker_2d_gui import Walker2DGUI
from walker_2d_logic import Walker2DEnvironment, Agent, PPO, TD3, SAC

def test_colors():
    """Test that multiple configurations get different colors"""
    root = tk.Tk()
    env = Walker2DEnvironment()
    policies = {'PPO': PPO, 'TD3': TD3, 'SAC': SAC}
    agent = Agent(env, PPO(env.get_state_dim(), env.get_action_dim(), [64, 64]))
    
    gui = Walker2DGUI(root, env, agent, policies)
    
    # Simulate multiple TD3 configurations with rewards data
    configs = [
        'TD3',
        'TD3_Expl_a0.00010_c0.00020',
        'TD3_TPS_a0.00015_c0.00025',
        'TD3_Expl_a0.00020_c0.00030',
        'TD3_TPS_a0.00025_c0.00035'
    ]
    
    # Add fake rewards data to simulate actual training
    for config in configs:
        gui.rewards_data[config] = [1.0, 2.0, 3.0]
    
    print("Testing color assignment for multiple configurations:")
    for config in configs:
        # Simulate color assignment as it happens in _update_plot
        if config not in gui.config_colors:
            base_method = config.split('_')[0]
            if config in gui.colors and len([k for k in gui.rewards_data.keys() if k.startswith(base_method)]) == 1:
                color = gui.colors[config]
            else:
                color_idx = len(gui.config_colors) % len(gui.color_palette)
                color = gui.color_palette[color_idx]
            gui.config_colors[config] = color
        
        print(f"  {config:35s} -> {gui.config_colors[config]}")
    
    # Check that all colors are unique
    colors_used = list(gui.config_colors.values())
    unique_colors = set(colors_used)
    
    if len(colors_used) == len(unique_colors):
        print(f"\n✓ All {len(colors_used)} configurations have unique colors!")
    else:
        print(f"\n✗ Warning: {len(colors_used)} configs but only {len(unique_colors)} unique colors")
    
    # Now test with only one config
    print("\nTesting single TD3 configuration (should use default color):")
    gui.rewards_data = {'TD3': [1.0, 2.0, 3.0]}
    gui.config_colors = {}
    
    config = 'TD3'
    if config not in gui.config_colors:
        base_method = config.split('_')[0]
        if config in gui.colors and len([k for k in gui.rewards_data.keys() if k.startswith(base_method)]) == 1:
            color = gui.colors[config]
        else:
            color_idx = len(gui.config_colors) % len(gui.color_palette)
            color = gui.color_palette[color_idx]
        gui.config_colors[config] = color
    
    print(f"  {config:35s} -> {gui.config_colors[config]}")
    if gui.config_colors[config] == gui.colors['TD3']:
        print("  ✓ Single config uses default TD3 color")
    
    env.close()
    root.destroy()

def test_animation_toggle():
    """Test animation toggle shows message"""
    print("\nTesting animation toggle:")
    root = tk.Tk()
    env = Walker2DEnvironment()
    policies = {'PPO': PPO, 'TD3': TD3, 'SAC': SAC}
    agent = Agent(env, PPO(env.get_state_dim(), env.get_action_dim(), [64, 64]))
    
    gui = Walker2DGUI(root, env, agent, policies)
    
    # Test enabling animation
    gui.anim_var.set(True)
    gui._toggle_animation()
    
    # Check if canvas has text
    canvas_items = gui.anim_canvas.find_all()
    if len(canvas_items) > 0:
        print("  ✓ Animation toggle shows message on canvas")
    else:
        print("  ✗ Animation canvas is empty")
    
    env.close()
    root.destroy()

if __name__ == '__main__':
    test_colors()
    test_animation_toggle()
    print("\n✓ All tests completed!")
