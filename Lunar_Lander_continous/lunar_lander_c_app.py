import tkinter as tk
from lunar_lander_c_logic import (
    LunarLanderCEnv,
    Agent,
    RainbowPolicy,
)
from lunar_lander_c_gui import LunarLanderCGUI


def main():
    env = LunarLanderCEnv(
        seed=None,
        render_mode=None,
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("LunarLander observation space must have a shape.")
    state_dim = int(obs_shape[0])
    action_dim = env.action_dim

    policy = RainbowPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        alpha=0.0003,
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
        grad_clip=0.5,
        target_update=100,
        priority_alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_steps=100000,
        atoms=51,
        vmin=-300.0,
        vmax=300.0,
        noisy_sigma=0.5,
        n_step=3,
    )

    agent = Agent(env, policy)

    root = tk.Tk()
    app = LunarLanderCGUI(root, env, agent, [policy])
    root.mainloop()


if __name__ == "__main__":
    main()
