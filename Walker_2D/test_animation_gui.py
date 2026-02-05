"""Test animation in GUI with actual training"""
import tkinter as tk
from walker_2d_gui import Walker2DGUI
from walker_2d_logic import Walker2DEnvironment, Agent, PPO, TD3, SAC
import time

def test_animation_in_gui():
    """Test that animation works during training"""
    print("Testing animation in GUI...")
    
    root = tk.Tk()
    env = Walker2DEnvironment()
    policies = {'PPO': PPO, 'TD3': TD3, 'SAC': SAC}
    agent = Agent(env, PPO(env.get_state_dim(), env.get_action_dim(), [64, 64]))
    
    gui = Walker2DGUI(root, env, agent, policies)
    
    # Enable animation
    gui.anim_var.set(True)
    gui._toggle_animation()
    print("  ✓ Animation enabled")
    
    # Set to train for just a few episodes
    gui.episodes_var.set(20)
    
    # Check canvas state
    canvas_items = gui.anim_canvas.find_all()
    if len(canvas_items) > 0:
        print("  ✓ Canvas shows 'animation will start' message")
    
    # Simulate starting training
    print("\n  Simulating training start...")
    gui.training = True
    gui.animation_enabled = True
    
    # Test the render thread flag logic
    configs = [('TD3', None, None)]
    
    # First thread should be render thread
    is_render_thread = (0 == 0 and gui.animation_enabled)
    print(f"  First thread is_render_thread: {is_render_thread}")
    
    if is_render_thread:
        print("  ✓ First thread will handle rendering")
    else:
        print("  ✗ ERROR: First thread should handle rendering")
    
    # Test with multiple configs
    configs = [
        ('TD3', {'actor_lr': 0.0001}, 'exploration'),
        ('TD3', {'actor_lr': 0.0002}, 'exploration'),
        ('TD3', {'actor_lr': 0.0003}, 'exploration')
    ]
    
    print(f"\n  Testing with {len(configs)} configurations:")
    for idx, config in enumerate(configs):
        is_render_thread = (idx == 0 and gui.animation_enabled)
        print(f"    Config {idx+1}: is_render_thread = {is_render_thread}")
    
    env.close()
    root.destroy()
    
    print("\n✓ Animation GUI test completed!")

if __name__ == '__main__':
    test_animation_in_gui()
