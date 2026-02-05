# Walker2D RL Application

## Goal:
- Create for reinforcement learning the app form gym library mujoco with the name `walker_2D` with the gym environment `Walker2d-v5`, to run live with selectable methods and comparable methods with a live reward plot.

## Very important: 
- !!! use `..\rl_workbench_init.md` for inital requirements !!!

## Concrete Features:
- Use Walker2d-v5 environment wrapper
- Use three RL algorithms: PPO, TD3, SAC

### Selectable Methods:
- **PPO** (Proximal Policy Optimization)
- **TD3** (Twin Delayed DDPG)
- **SAC** (Soft Actor-Critic)

### Scalable Method params:
- **TD3**: Learning rate for actor (actor_lr) with value-from, value-to, step.
- **TD3**: Learning rate for critic (critic_lr) with value-from, value-to, step.
- **TD3**: Selectable noise (Exploration Noise or Target Policy Smoothing Noise)

### Selectable Environment Parameters (Walker-2D):
- forward_reward_weight (default: 1.0)
- ctrl_cost_weight (default: 0.001)
- healthy_reward (default 1.0)
- terminale_when_unhealthy (default True)
- healthy_z_range (default (0.8, 2))
- healthy_angle_range (default (-1, 1))
- reset_noise_scale (default: 0.005)
