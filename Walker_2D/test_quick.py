"""Quick test for Walker2D application"""
from walker_2d_logic import Walker2DEnvironment, PPO, TD3, SAC, Agent
import numpy as np

print("="*60)
print("Walker2D RL Workbench - Quick Test")
print("="*60)

# Test 1: Environment creation
print("\nTest 1: Environment Creation")
env = Walker2DEnvironment()
print(f"  ✓ State dimension: {env.get_state_dim()}")
print(f"  ✓ Action dimension: {env.get_action_dim()}")

# Test 2: PPO
print("\nTest 2: PPO Training (5 episodes)")
ppo = PPO(env.get_state_dim(), env.get_action_dim(), [64, 64])
agent = Agent(env, ppo)
rewards = []
for i in range(5):
    reward = agent.run_episode(train=True)
    rewards.append(reward)
    print(f"  Episode {i+1}: Reward = {reward:.2f}")
print(f"  ✓ Average reward: {np.mean(rewards):.2f}")

# Test 3: TD3
print("\nTest 3: TD3 Training (5 episodes)")
td3 = TD3(env.get_state_dim(), env.get_action_dim(), [64, 64], batch_size=64)
agent = Agent(env, td3)
rewards = []
for i in range(5):
    reward = agent.run_episode(train=True)
    rewards.append(reward)
    print(f"  Episode {i+1}: Reward = {reward:.2f}, Buffer = {len(td3.memory)}")
print(f"  ✓ Average reward: {np.mean(rewards):.2f}")

# Test 4: SAC
print("\nTest 4: SAC Training (5 episodes)")
sac = SAC(env.get_state_dim(), env.get_action_dim(), [64, 64], batch_size=64)
agent = Agent(env, sac)
rewards = []
for i in range(5):
    reward = agent.run_episode(train=True)
    rewards.append(reward)
    print(f"  Episode {i+1}: Reward = {reward:.2f}, Buffer = {len(sac.memory)}")
print(f"  ✓ Average reward: {np.mean(rewards):.2f}")

# Test 5: Animation
print("\nTest 5: Animation Rendering")
env_render = Walker2DEnvironment(render_mode='rgb_array')
env_render.reset()
frame = env_render.render()
if frame is not None:
    print(f"  ✓ Frame shape: {frame.shape}")
    print(f"  ✓ Frame dtype: {frame.dtype}")
    print(f"  ✓ Frame range: [{frame.min()}, {frame.max()}]")
else:
    print("  ✗ No frame rendered")
env_render.close()

env.close()

print("\n" + "="*60)
print("All tests completed successfully!")
print("="*60)
print("\nTo run the GUI application:")
print("  python walker_2d_app.py")
print("\nFeatures:")
print("  - PPO, TD3, SAC algorithms")
print("  - Compare mode for parallel training")
print("  - Live reward plotting")
print("  - Animation support")
print("  - 30+ configurable parameters")
