import tkinter as tk

from mountain_car_gui import MountainCarGUI
from mountain_car_logic import (
    Agent,
    MountainCarEnv,
    DQNPolicy,
    DDQNPolicy,
    PrioritizedDDQNPolicy,
    DuelingDDQNPolicy,
    NoisyDQNPolicy,
    DistributionalDQNPolicy,
    RainbowPolicy,
)


def main():
    env = MountainCarEnv(env_name="MountainCarContinuous-v0", goal_velocity=0.0, action_bins=11)
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        env.close()
        raise ValueError("MountainCar observation space must be a Box with shape.")
    state_dim = int(obs_shape[0])
    action_dim = env.action_bins
    policies = [
        DQNPolicy(state_dim, action_dim, alpha=0.0005, gamma=0.99, eps_start=1.0, eps_end=0.02, eps_decay=0.0005, hidden_layers=2, hidden_units=128, batch_size=128, replay_size=50000, activation="ReLU", warmup_steps=2000, normalize_states=True, reward_scale=1.0, grad_clip=10.0),
        DDQNPolicy(state_dim, action_dim, alpha=0.0005, gamma=0.99, eps_start=1.0, eps_end=0.02, eps_decay=0.0005, hidden_layers=2, hidden_units=128, batch_size=128, replay_size=50000, activation="ReLU", warmup_steps=2000, normalize_states=True, reward_scale=1.0, grad_clip=10.0),
        PrioritizedDDQNPolicy(state_dim, action_dim, alpha=0.0005, gamma=0.99, eps_start=1.0, eps_end=0.02, eps_decay=0.0005, hidden_layers=2, hidden_units=128, batch_size=128, replay_size=50000, activation="ReLU", warmup_steps=2000, normalize_states=True, reward_scale=1.0, grad_clip=10.0),
        DuelingDDQNPolicy(state_dim, action_dim, alpha=0.0005, gamma=0.99, eps_start=1.0, eps_end=0.02, eps_decay=0.0005, hidden_layers=2, hidden_units=128, batch_size=128, replay_size=50000, activation="ReLU", warmup_steps=2000, normalize_states=True, reward_scale=1.0, grad_clip=10.0),
        DistributionalDQNPolicy(state_dim, action_dim, alpha=0.0005, gamma=0.99, eps_start=1.0, eps_end=0.02, eps_decay=0.0005, hidden_layers=2, hidden_units=128, batch_size=128, replay_size=50000, activation="ReLU", warmup_steps=2000, normalize_states=True, reward_scale=1.0, grad_clip=10.0),
        NoisyDQNPolicy(state_dim, action_dim, alpha=0.0005, gamma=0.99, eps_start=1.0, eps_end=0.02, eps_decay=0.0005, hidden_layers=2, hidden_units=128, batch_size=128, replay_size=50000, activation="ReLU", warmup_steps=2000, normalize_states=True, reward_scale=1.0, grad_clip=10.0),
        RainbowPolicy(state_dim, action_dim, alpha=0.0005, gamma=0.99, eps_start=1.0, eps_end=0.02, eps_decay=0.0005, hidden_layers=2, hidden_units=128, batch_size=128, replay_size=50000, activation="ReLU", warmup_steps=2000, normalize_states=True, reward_scale=1.0, grad_clip=10.0),
    ]
    agent = Agent(env, policies[0])
    root = tk.Tk()
    app = MountainCarGUI(root, env, agent, policies)
    root.mainloop()


if __name__ == "__main__":
    main()
