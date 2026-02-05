"""
Compare SAC performance with different training frequencies
"""

from bipedal_walker_logic import BipedalWalkerEnvironment, SAC, Agent
import time

print("SAC Performance Comparison")
print("=" * 60)

test_configs = [
    {'train_freq': 2, 'gradient_steps': 1, 'name': 'Freq=2, GradSteps=1'},
    {'train_freq': 4, 'gradient_steps': 1, 'name': 'Freq=4, GradSteps=1 (default)'},
    {'train_freq': 8, 'gradient_steps': 1, 'name': 'Freq=8, GradSteps=1'},
    {'train_freq': 4, 'gradient_steps': 2, 'name': 'Freq=4, GradSteps=2'},
]

episodes = 5

for config in test_configs:
    env = BipedalWalkerEnvironment()
    policy = SAC(
        state_dim=24, 
        action_dim=4, 
        hidden_dims=[64, 64],
        batch_size=64,
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps']
    )
    agent = Agent(env, policy)
    
    print(f"\nTesting: {config['name']}")
    print("-" * 60)
    
    start_time = time.time()
    
    for ep in range(episodes):
        reward = agent.run_episode(train=True)
    
    elapsed = time.time() - start_time
    
    returns = agent.get_returns()
    avg_reward = sum(returns) / len(returns)
    
    print(f"  Time:         {elapsed:.1f}s ({elapsed/episodes:.1f}s per episode)")
    print(f"  Avg Reward:   {avg_reward:.2f}")
    print(f"  Memory Size:  {len(policy.memory)}")
    print(f"  Total Steps:  {policy.total_steps}")
    
    env.close()

print("\n" + "=" * 60)
print("Recommendation:")
print("  - For SPEED:     Use train_freq=8, gradient_steps=1")
print("  - For BALANCE:   Use train_freq=4, gradient_steps=1 (default)")
print("  - For LEARNING:  Use train_freq=2, gradient_steps=2")
