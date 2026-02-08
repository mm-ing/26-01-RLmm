# Walker2D RL Application

## Goal:
- Create for reinforcement learning the app form gym library mujoco with the name `pusher` with the gym mujoco environment `Pusher-v5`, to run live with selectable methods and comparable methods with a live reward plot.

## Very important: 
- !!! use `..\rl_workbench_init.md` for inital requirements !!!

## Concrete Features:
- Use Pusher-v5 environment wrapper
- Use three RL algorithms: DDPG, TD3, SAC

### Selectable Methods:
- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed DDPG)
- **SAC** (Soft Actor-Critic)

### Scalable Method params:
- **SAC**: alpha (Verhältnis Exploration und Exploitation) with value-from, value-to, step.
- **SAC**: Replay butter size with value-from, value-to, step.
- **SAC**: gamma (discount factor)

### Selectable Environmental Parameters (Pusher):
- reward_near_weight (default: 0.5)
- reward_dist_weight (default: 1)
- reward_control_weight (default 0.1)
