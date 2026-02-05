"""
Quick test to verify SAC is learning
"""

from bipedal_walker_logic import BipedalWalkerEnvironment, SAC, Agent
import numpy as np

print("Testing SAC Learning...")
print("="*50)

# Create environment
env = BipedalWalkerEnvironment(hardcore=False)

# Create SAC policy with optimized settings
policy = SAC(
    state_dim=24,
    action_dim=4,
    hidden_dims=[256, 256],
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    batch_size=64,
    train_freq=4,  # Train every 4 steps for good balance
    gradient_steps=1  # 1 gradient update per training
)

# Create agent
agent = Agent(env, policy)

# Train for 20 episodes
print("\nTraining SAC for 20 episodes...")
for ep in range(20):
    reward = agent.run_episode(render=False, train=True)
    memory = len(policy.memory)
    
    # Print every episode for first 5, then every 5
    if ep < 5 or (ep + 1) % 5 == 0:
        recent_rewards = agent.get_returns()[-5:] if len(agent.get_returns()) >= 5 else agent.get_returns()
        avg_reward = np.mean(recent_rewards)
        print(f"Episode {ep+1:2d}: Reward={reward:7.2f}, Avg(last {len(recent_rewards)})={avg_reward:7.2f}, Memory={memory:5d}")

print("\n" + "="*50)
print("Training Complete!")

# Get all returns
returns = agent.get_returns()
print(f"\nFirst 5 rewards: {[f'{r:.2f}' for r in returns[:5]]}")
print(f"Last 5 rewards:  {[f'{r:.2f}' for r in returns[-5:]]}")
print(f"Average first 5: {np.mean(returns[:5]):.2f}")
print(f"Average last 5:  {np.mean(returns[-5:]):.2f}")

# Check if learning is happening
improvement = np.mean(returns[-5:]) - np.mean(returns[:5])
print(f"\nImprovement: {improvement:.2f}")

if improvement > 0:
    print("✓ SAC is learning! Rewards are improving.")
else:
    print("Note: Learning may need more episodes or hyperparameter tuning.")

env.close()
