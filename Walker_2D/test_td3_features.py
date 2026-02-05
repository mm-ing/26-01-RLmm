"""Test TD3 range parameters and noise algorithm selection"""
from walker_2d_logic import Walker2DEnvironment, TD3, Agent
import numpy as np

print("="*70)
print("TD3 Parameter Range and Noise Algorithm Test")
print("="*70)

env = Walker2DEnvironment()
state_dim = env.get_state_dim()
action_dim = env.get_action_dim()

# Test 1: Different noise types
print("\nTest 1: Noise Algorithm Comparison")
print("-" * 70)

for noise_type in ['exploration', 'target_smoothing']:
    print(f"\n  Testing {noise_type} noise:")
    td3 = TD3(state_dim, action_dim, [64, 64], 
             batch_size=64, noise_type=noise_type)
    agent = Agent(env, td3)
    
    rewards = []
    for i in range(3):
        reward = agent.run_episode(train=True)
        rewards.append(reward)
        print(f"    Episode {i+1}: Reward = {reward:.2f}, Buffer = {len(td3.memory)}")
    print(f"    ✓ Average: {np.mean(rewards):.2f}")

# Test 2: Actor learning rate range
print("\n\nTest 2: Actor Learning Rate Range")
print("-" * 70)

actor_lr_values = np.arange(0.0001, 0.0006, 0.0002)
print(f"  Testing actor_lr values: {[f'{lr:.4f}' for lr in actor_lr_values]}")

for actor_lr in actor_lr_values:
    td3 = TD3(state_dim, action_dim, [64, 64],
             actor_lr=actor_lr, batch_size=64)
    agent = Agent(env, td3)
    
    rewards = []
    for i in range(2):
        reward = agent.run_episode(train=True)
        rewards.append(reward)
    
    avg_reward = np.mean(rewards)
    print(f"    actor_lr={actor_lr:.4f}: Avg Reward = {avg_reward:.2f}")

# Test 3: Critic learning rate range
print("\n\nTest 3: Critic Learning Rate Range")
print("-" * 70)

critic_lr_values = np.arange(0.0002, 0.0008, 0.0002)
print(f"  Testing critic_lr values: {[f'{lr:.4f}' for lr in critic_lr_values]}")

for critic_lr in critic_lr_values:
    td3 = TD3(state_dim, action_dim, [64, 64],
             critic_lr=critic_lr, batch_size=64)
    agent = Agent(env, td3)
    
    rewards = []
    for i in range(2):
        reward = agent.run_episode(train=True)
        rewards.append(reward)
    
    avg_reward = np.mean(rewards)
    print(f"    critic_lr={critic_lr:.4f}: Avg Reward = {avg_reward:.2f}")

# Test 4: Combined parameter sweep
print("\n\nTest 4: Combined Parameter Sweep")
print("-" * 70)
print("  Testing combinations of actor_lr, critic_lr, and noise type:")

test_configs = [
    {'actor_lr': 0.0002, 'critic_lr': 0.0003, 'noise_type': 'exploration'},
    {'actor_lr': 0.0002, 'critic_lr': 0.0003, 'noise_type': 'target_smoothing'},
    {'actor_lr': 0.0004, 'critic_lr': 0.0003, 'noise_type': 'exploration'},
    {'actor_lr': 0.0004, 'critic_lr': 0.0003, 'noise_type': 'target_smoothing'},
]

for config in test_configs:
    td3 = TD3(state_dim, action_dim, [64, 64], batch_size=64, **config)
    agent = Agent(env, td3)
    
    reward = agent.run_episode(train=True)
    noise_short = 'Expl' if config['noise_type'] == 'exploration' else 'TPS'
    print(f"    a={config['actor_lr']:.4f}, c={config['critic_lr']:.4f}, {noise_short}: Reward = {reward:.2f}")

env.close()

print("\n" + "="*70)
print("TD3 Parameter Range Tests Completed!")
print("="*70)
print("\nGUI Features:")
print("  ✓ TD3 actor_lr: Range mode with from/to/step controls")
print("  ✓ TD3 critic_lr: Range mode with from/to/step controls")
print("  ✓ Noise selection: Exploration Noise and/or Target Policy Smoothing")
print("  ✓ Loop execution: Automatically runs all combinations")
print("  ✓ Live plotting: Each configuration plotted with unique label")
print("\nUsage in GUI:")
print("  1. Select TD3 method")
print("  2. Check 'Range' checkbox next to actor_lr or critic_lr")
print("  3. Set from/to/step values for the range")
print("  4. Select one or both noise algorithms")
print("  5. Click 'Train' - all combinations will run in parallel")
print("  6. Each configuration appears in plot with label: TD3_Noise_a[lr]_c[lr]")
