# Goal:
- Create for reinforcement learning the app with the name `mountain_car` with the gym environment `MountainCarContinuous-v0`, to run live with selectable methods and compare methods with a live reward plot.

## Very important: 
- !!! use `..\rl_init.md` for inital requirements !!!

## Additional Features
- Selectable methods (dropdown): DQN, DDQN, Prioritized DDQN, Dueling DDQN, Distributional DQN, Noisy DQN, Rainbow
- Selectable goal_velocity (default: 0.0)
- Selectable parameters: seed and as options: x_init (default: np.pi), y_init (default: 1.0)