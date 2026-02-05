"""Test animation rendering"""
from half_cheetah_logic import HalfCheetahEnvironment
import numpy as np

# Test 1: Check if render returns a valid frame
print("Test 1: Checking render with rgb_array mode...")
env = HalfCheetahEnvironment(render_mode='rgb_array')
env.reset()

frame = env.render()
if frame is not None:
    print(f"  ✓ Frame rendered successfully")
    print(f"  ✓ Frame shape: {frame.shape}")
    print(f"  ✓ Frame dtype: {frame.dtype}")
    print(f"  ✓ Frame min/max: {frame.min()}/{frame.max()}")
else:
    print("  ✗ No frame returned!")

# Test 2: Check frame after a step
print("\nTest 2: Checking frame after step...")
action = np.zeros(env.get_action_dim())
env.step(action)
frame2 = env.render()
if frame2 is not None:
    print(f"  ✓ Frame after step rendered")
    print(f"  ✓ Frame shape: {frame2.shape}")
else:
    print("  ✗ No frame after step!")

env.close()

# Test 3: Check render mode None
print("\nTest 3: Checking render with None mode...")
env2 = HalfCheetahEnvironment(render_mode=None)
env2.reset()
frame3 = env2.render()
if frame3 is None:
    print("  ✓ Correctly returns None when render_mode=None")
else:
    print("  ✗ Should return None but got frame!")

env2.close()

print("\n" + "="*50)
print("Animation rendering tests completed!")
print("="*50)
