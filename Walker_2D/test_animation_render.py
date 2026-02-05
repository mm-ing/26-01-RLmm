"""Quick test for animation rendering"""
from walker_2d_logic import Walker2DEnvironment, PPO, Agent

print("Testing animation rendering...")

# Test 1: Environment with render_mode
print("\n1. Testing environment with render_mode='rgb_array':")
env = Walker2DEnvironment(render_mode='rgb_array')
state_dim = env.get_state_dim()
action_dim = env.get_action_dim()

policy = PPO(state_dim, action_dim, [64, 64])
agent = Agent(env, policy)

print(f"   Environment created with render_mode: {env.render_mode}")

# Run a few steps and check if rendering works
state = env.reset()
for i in range(10):
    action, _ = policy.select_action(state)
    state, reward, done, truncated, info = env.step(action)
    
    # Try to render
    frame = env.render()
    if frame is not None:
        print(f"   Step {i+1}: Frame shape = {frame.shape}, dtype = {frame.dtype}")
        if i == 0:
            print(f"   ✓ Rendering works! Frame is {frame.shape[0]}x{frame.shape[1]} RGB array")
        break
    else:
        print(f"   Step {i+1}: Frame is None")

if frame is None:
    print("   ✗ Rendering failed - all frames were None")

env.close()

# Test 2: Environment without render_mode
print("\n2. Testing environment without render_mode:")
env2 = Walker2DEnvironment(render_mode=None)
print(f"   Environment created with render_mode: {env2.render_mode}")

state = env2.reset()
frame = env2.render()
if frame is None:
    print(f"   ✓ Correct: Frame is None when render_mode is None")
else:
    print(f"   Frame: {frame}")

env2.close()

print("\n✓ Animation rendering test completed!")
