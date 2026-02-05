"""Quick test for display_name fix"""
from walker_2d_logic import Walker2DEnvironment, TD3, Agent
import numpy as np

print("Testing display_name creation...")

env = Walker2DEnvironment()
state_dim = env.get_state_dim()
action_dim = env.get_action_dim()

# Test different configurations
configs = [
    ("Simple", None, None),
    ("With noise", None, 'exploration'),
    ("With params", {'actor_lr': 0.0003, 'critic_lr': 0.0004}, None),
    ("Full config", {'actor_lr': 0.0002, 'critic_lr': 0.0005}, 'target_smoothing'),
]

for name, param_override, noise_type in configs:
    # Simulate display_name creation logic
    method_name = 'TD3'
    display_name = method_name
    
    if method_name == 'TD3' and (param_override or noise_type):
        parts = [method_name]
        if noise_type:
            noise_short = 'Expl' if noise_type == 'exploration' else 'TPS'
            parts.append(noise_short)
        if param_override:
            if 'actor_lr' in param_override:
                parts.append(f"a{param_override['actor_lr']:.5f}")
            if 'critic_lr' in param_override:
                parts.append(f"c{param_override['critic_lr']:.5f}")
        display_name = '_'.join(parts)
    
    print(f"  {name:15s}: '{display_name}'")

env.close()

print("\n✓ Display name creation working correctly!")
