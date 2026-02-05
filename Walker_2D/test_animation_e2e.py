"""End-to-end animation test with actual training"""
from walker_2d_logic import Walker2DEnvironment, PPO, Agent
import numpy as np

print("End-to-end animation test...")
print("=" * 60)

# Create environment with rendering enabled
print("\n1. Creating environment with render_mode='rgb_array'")
env = Walker2DEnvironment(render_mode='rgb_array')
state_dim = env.get_state_dim()
action_dim = env.get_action_dim()

print(f"   State dim: {state_dim}, Action dim: {action_dim}")
print(f"   Render mode: {env.render_mode}")

# Create policy and agent
policy = PPO(state_dim, action_dim, [64, 64])
agent = Agent(env, policy)

print("\n2. Training for 10 episodes with animation...")
frames_captured = 0

for episode in range(10):
    # Run episode
    reward = agent.run_episode(train=True, render=False)
    
    # Simulate what the GUI does - render every 5 episodes
    if episode % 5 == 0:
        # After episode, get a frame
        env.reset()  # Reset to get a frame
        frame = env.render()
        
        if frame is not None:
            frames_captured += 1
            print(f"   Episode {episode:2d}: Reward={reward:7.2f}, Frame={frame.shape} ✓")
        else:
            print(f"   Episode {episode:2d}: Reward={reward:7.2f}, Frame=None ✗")
    else:
        print(f"   Episode {episode:2d}: Reward={reward:7.2f}")

print(f"\n3. Results:")
print(f"   Total episodes: 10")
print(f"   Frames captured: {frames_captured} (expected 2: episodes 0 and 5)")
print(f"   Average reward: {np.mean(agent.rewards_history):.2f}")

if frames_captured >= 2:
    print("\n✓ Animation test PASSED - frames successfully captured during training!")
else:
    print("\n✗ Animation test FAILED - frames not captured")

env.close()
