import tkinter as tk
from lunar_lander_logic import (
    LunarLanderEnv,
    Agent,
    DQNPolicy,
)
from lunar_lander_gui import LunarLanderGUI


def main():
    env = LunarLanderEnv(
        seed=None,
        render_mode=None,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("LunarLander observation space must have a shape.")
    state_dim = int(obs_shape[0])
    action_dim = env.total_actions

    policy = DQNPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        alpha=0.0005,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.001,
        hidden_layers=2,
        hidden_units=256,
        batch_size=64,
        replay_size=100000,
        activation="ReLU",
        warmup_steps=1000,
        grad_clip=10.0,
        target_update=100,
    )

    agent = Agent(env, policy)

    root = tk.Tk()
    app = LunarLanderGUI(root, env, agent, [policy])
    root.mainloop()


if __name__ == "__main__":
    main()
