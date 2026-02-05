import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from lunar_lander_c_logic import (
    Agent,
    LunarLanderCEnv,
    RainbowPolicy,
    A2CPolicy,
    TRPOPolicy,
    PPOPolicy,
)


@dataclass
class MethodConfig:
    name: str
    policy_cls: object
    color_light: str
    color_bold: str


class LunarLanderCGUI:
    def __init__(self, master: tk.Tk, env: LunarLanderCEnv, agent: Agent, policies: List):
        self.master = master
        self.env = env
        self.agent = agent
        self.policies = policies
        self.master.title("Lunar Lander Continuous RL Demo")

        self.methods = [
            MethodConfig("Rainbow", RainbowPolicy, "#ffb3e6", "#ff4db8"),
            MethodConfig("A2C", A2CPolicy, "#ffb3b3", "#ff4d4d"),
            MethodConfig("TRPO", TRPOPolicy, "#b3e6ff", "#3399ff"),
            MethodConfig("PPO", PPOPolicy, "#b3ffb3", "#4dff4d"),
        ]

        self.returns_by_method: Dict[str, List[float]] = {}
        self.compare_methods = tk.BooleanVar(value=True)
        self.method_var = tk.StringVar(value=self.methods[0].name)
        self._render_method: Optional[str] = None

        self._workers: Dict[str, Dict] = {}
        self._queue: queue.Queue = queue.Queue()
        self._render_image: Optional[ImageTk.PhotoImage] = None
        self._animation_enabled = True

        self._build_ui()
        self._schedule_queue_poll()

    def _build_ui(self):
        top_frame = ttk.Frame(self.master)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=8, pady=6)

        control_frame = ttk.LabelFrame(self.master, text="Training Configuration")
        control_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

        method_frame = ttk.LabelFrame(self.master, text="Method Parameters")
        method_frame.grid(row=1, column=1, sticky="nsew", padx=8, pady=6)

        action_frame = ttk.LabelFrame(self.master, text="Action")
        action_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=8, pady=6)

        plot_frame = ttk.Frame(self.master)
        plot_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=8, pady=6)

        self.render_canvas = tk.Canvas(top_frame, width=600, height=400, bg="black")
        self.render_canvas.pack(fill="both", expand=True)

        # Column 1
        ttk.Label(control_frame, text="Episodes").grid(row=0, column=0, sticky="w")
        self.episodes_var = tk.IntVar(value=500)
        ttk.Entry(control_frame, textvariable=self.episodes_var, width=8).grid(row=0, column=1)

        ttk.Label(control_frame, text="Alpha").grid(row=1, column=0, sticky="w")
        self.alpha_var = tk.DoubleVar(value=0.0003)
        ttk.Entry(control_frame, textvariable=self.alpha_var, width=8).grid(row=1, column=1)

        ttk.Label(control_frame, text="Gamma").grid(row=2, column=0, sticky="w")
        self.gamma_var = tk.DoubleVar(value=0.99)
        ttk.Entry(control_frame, textvariable=self.gamma_var, width=8).grid(row=2, column=1)

        ttk.Label(control_frame, text="Step delay (s)").grid(row=3, column=0, sticky="w")
        self.step_delay_var = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.step_delay_var, width=8).grid(row=3, column=1)

        ttk.Label(control_frame, text="Max steps").grid(row=4, column=0, sticky="w")
        self.max_steps_var = tk.IntVar(value=1000)
        ttk.Entry(control_frame, textvariable=self.max_steps_var, width=8).grid(row=4, column=1)

        ttk.Label(control_frame, text="Seed").grid(row=5, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(control_frame, textvariable=self.seed_var, width=8).grid(row=5, column=1)

        ttk.Label(control_frame, text="Render interval (s)").grid(row=6, column=0, sticky="w")
        self.render_interval_var = tk.DoubleVar(value=0.05)
        ttk.Entry(control_frame, textvariable=self.render_interval_var, width=8).grid(row=6, column=1)

        ttk.Label(control_frame, text="Continuous").grid(row=7, column=0, sticky="w")
        self.continuous_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, variable=self.continuous_var).grid(row=7, column=1, sticky="w")

        ttk.Label(control_frame, text="Gravity").grid(row=8, column=0, sticky="w")
        self.gravity_var = tk.DoubleVar(value=-10.0)
        ttk.Entry(control_frame, textvariable=self.gravity_var, width=8).grid(row=8, column=1)

        ttk.Label(control_frame, text="Enable wind").grid(row=9, column=0, sticky="w")
        self.enable_wind_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, variable=self.enable_wind_var).grid(row=9, column=1, sticky="w")

        ttk.Label(control_frame, text="Wind power").grid(row=10, column=0, sticky="w")
        self.wind_power_var = tk.DoubleVar(value=15.0)
        ttk.Entry(control_frame, textvariable=self.wind_power_var, width=8).grid(row=10, column=1)

        ttk.Label(control_frame, text="Turbulence power").grid(row=11, column=0, sticky="w")
        self.turbulence_power_var = tk.DoubleVar(value=1.5)
        ttk.Entry(control_frame, textvariable=self.turbulence_power_var, width=8).grid(row=11, column=1)

        ttk.Label(control_frame, text="Grad clip").grid(row=12, column=0, sticky="w")
        self.grad_clip_var = tk.DoubleVar(value=1.0)
        ttk.Entry(control_frame, textvariable=self.grad_clip_var, width=8).grid(row=12, column=1)

        # Column 2
        ttk.Label(control_frame, text="Hidden layers").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.hidden_layers_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.hidden_layers_var, width=8).grid(row=0, column=3)

        ttk.Label(control_frame, text="Hidden units").grid(row=1, column=2, sticky="w", padx=(12, 0))
        self.hidden_units_var = tk.IntVar(value=256)
        ttk.Entry(control_frame, textvariable=self.hidden_units_var, width=8).grid(row=1, column=3)

        ttk.Label(control_frame, text="Batch size").grid(row=2, column=2, sticky="w", padx=(12, 0))
        self.batch_size_var = tk.IntVar(value=64)
        ttk.Entry(control_frame, textvariable=self.batch_size_var, width=8).grid(row=2, column=3)

        ttk.Label(control_frame, text="N steps").grid(row=3, column=2, sticky="w", padx=(12, 0))
        self.n_steps_var = tk.IntVar(value=512)
        ttk.Entry(control_frame, textvariable=self.n_steps_var, width=8).grid(row=3, column=3)

        ttk.Label(control_frame, text="Activation").grid(row=4, column=2, sticky="w", padx=(12, 0))
        self.activation_var = tk.StringVar(value="ReLU")
        ttk.Combobox(
            control_frame,
            values=["ReLU", "LeakyReLU", "Tanh", "ELU", "GELU"],
            textvariable=self.activation_var,
            state="readonly",
            width=8,
        ).grid(row=4, column=3, sticky="w")

        ttk.Label(control_frame, text="Method").grid(row=5, column=2, sticky="w", padx=(12, 0))
        ttk.Combobox(control_frame, values=[m.name for m in self.methods], textvariable=self.method_var, state="readonly").grid(row=5, column=3, sticky="w")

        ttk.Checkbutton(control_frame, text="Compare methods", variable=self.compare_methods).grid(row=6, column=2, columnspan=2, sticky="w", padx=(12, 0))

        # Method-specific parameters
        rainbow_frame = ttk.LabelFrame(method_frame, text="Rainbow")
        rainbow_frame.grid(row=0, column=0, padx=6, pady=4, sticky="nsew")
        ttk.Label(rainbow_frame, text="Replay size").grid(row=0, column=0, sticky="w")
        self.rainbow_replay_size_var = tk.IntVar(value=100000)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_replay_size_var, width=8).grid(row=0, column=1)
        ttk.Label(rainbow_frame, text="Target update").grid(row=1, column=0, sticky="w")
        self.rainbow_target_update_var = tk.IntVar(value=100)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_target_update_var, width=8).grid(row=1, column=1)
        ttk.Label(rainbow_frame, text="Atoms").grid(row=2, column=0, sticky="w")
        self.rainbow_atoms_var = tk.IntVar(value=51)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_atoms_var, width=8).grid(row=2, column=1)
        ttk.Label(rainbow_frame, text="Vmin").grid(row=3, column=0, sticky="w")
        self.rainbow_vmin_var = tk.DoubleVar(value=-300.0)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_vmin_var, width=8).grid(row=3, column=1)
        ttk.Label(rainbow_frame, text="Vmax").grid(row=4, column=0, sticky="w")
        self.rainbow_vmax_var = tk.DoubleVar(value=300.0)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_vmax_var, width=8).grid(row=4, column=1)
        ttk.Label(rainbow_frame, text="Noisy sigma").grid(row=5, column=0, sticky="w")
        self.rainbow_noisy_sigma_var = tk.DoubleVar(value=0.5)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_noisy_sigma_var, width=8).grid(row=5, column=1)

        a2c_frame = ttk.LabelFrame(method_frame, text="A2C")
        a2c_frame.grid(row=0, column=1, padx=6, pady=4, sticky="nsew")
        ttk.Label(a2c_frame, text="Value coef").grid(row=0, column=0, sticky="w")
        self.a2c_value_coef_var = tk.DoubleVar(value=0.5)
        ttk.Entry(a2c_frame, textvariable=self.a2c_value_coef_var, width=8).grid(row=0, column=1)
        ttk.Label(a2c_frame, text="Entropy coef").grid(row=1, column=0, sticky="w")
        self.a2c_entropy_coef_var = tk.DoubleVar(value=0.05)
        ttk.Entry(a2c_frame, textvariable=self.a2c_entropy_coef_var, width=8).grid(row=1, column=1)
        ttk.Label(a2c_frame, text="N steps").grid(row=2, column=0, sticky="w")
        self.a2c_n_steps_var = tk.IntVar(value=256)
        ttk.Entry(a2c_frame, textvariable=self.a2c_n_steps_var, width=8).grid(row=2, column=1)

        trpo_frame = ttk.LabelFrame(method_frame, text="TRPO")
        trpo_frame.grid(row=1, column=0, padx=6, pady=4, sticky="nsew")
        ttk.Label(trpo_frame, text="Max KL").grid(row=0, column=0, sticky="w")
        self.trpo_max_kl_var = tk.DoubleVar(value=0.01)
        ttk.Entry(trpo_frame, textvariable=self.trpo_max_kl_var, width=8).grid(row=0, column=1)
        ttk.Label(trpo_frame, text="Damping").grid(row=1, column=0, sticky="w")
        self.trpo_damping_var = tk.DoubleVar(value=0.1)
        ttk.Entry(trpo_frame, textvariable=self.trpo_damping_var, width=8).grid(row=1, column=1)
        ttk.Label(trpo_frame, text="GAE lambda").grid(row=2, column=0, sticky="w")
        self.trpo_gae_lambda_var = tk.DoubleVar(value=0.95)
        ttk.Entry(trpo_frame, textvariable=self.trpo_gae_lambda_var, width=8).grid(row=2, column=1)

        ppo_frame = ttk.LabelFrame(method_frame, text="PPO")
        ppo_frame.grid(row=1, column=1, padx=6, pady=4, sticky="nsew")
        ttk.Label(ppo_frame, text="Clip epsilon").grid(row=0, column=0, sticky="w")
        self.ppo_clip_epsilon_var = tk.DoubleVar(value=0.2)
        ttk.Entry(ppo_frame, textvariable=self.ppo_clip_epsilon_var, width=8).grid(row=0, column=1)
        ttk.Label(ppo_frame, text="Value coef").grid(row=1, column=0, sticky="w")
        self.ppo_value_coef_var = tk.DoubleVar(value=0.5)
        ttk.Entry(ppo_frame, textvariable=self.ppo_value_coef_var, width=8).grid(row=1, column=1)
        ttk.Label(ppo_frame, text="Entropy coef").grid(row=2, column=0, sticky="w")
        self.ppo_entropy_coef_var = tk.DoubleVar(value=0.05)
        ttk.Entry(ppo_frame, textvariable=self.ppo_entropy_coef_var, width=8).grid(row=2, column=1)
        ttk.Label(ppo_frame, text="N epochs").grid(row=3, column=0, sticky="w")
        self.ppo_n_epochs_var = tk.IntVar(value=10)
        ttk.Entry(ppo_frame, textvariable=self.ppo_n_epochs_var, width=8).grid(row=3, column=1)
        ttk.Label(ppo_frame, text="GAE lambda").grid(row=4, column=0, sticky="w")
        self.ppo_gae_lambda_var = tk.DoubleVar(value=0.95)
        ttk.Entry(ppo_frame, textvariable=self.ppo_gae_lambda_var, width=8).grid(row=4, column=1)

        # Action buttons
        self.train_button = ttk.Button(action_frame, text="Train and run", command=self._on_train)
        self.train_button.grid(row=0, column=0, padx=4, pady=4)

        self.cancel_button = ttk.Button(action_frame, text="Cancel learning", command=self._on_cancel, state="disabled")
        self.cancel_button.grid(row=0, column=1, padx=4, pady=4)

        self.reset_button = ttk.Button(action_frame, text="Reset", command=self._on_reset)
        self.reset_button.grid(row=0, column=2, padx=4, pady=4)

        self.save_button = ttk.Button(action_frame, text="Save plot", command=self._on_save_plot)
        self.save_button.grid(row=0, column=3, padx=4, pady=4)

        self.animation_button = ttk.Button(action_frame, text="Disable animation", command=self._toggle_animation)
        self.animation_button.grid(row=0, column=4, padx=4, pady=4)

        # Plot
        self.fig = Figure(figsize=(10, 4), facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, facecolor='#1e1e1e')
        self.ax.set_xlabel('Episode', color='white')
        self.ax.set_ylabel('Return', color='white')
        self.ax.set_title('Training Returns', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('#2b2b2b')
        self.ax.spines['right'].set_color('#2b2b2b')
        self.ax.grid(True, alpha=0.2, color='white')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.master.rowconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=0)
        self.master.rowconfigure(2, weight=0)
        self.master.rowconfigure(3, weight=2)
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)

    def _toggle_animation(self):
        self._animation_enabled = not self._animation_enabled
        if self._animation_enabled:
            self.animation_button.config(text="Disable animation")
        else:
            self.animation_button.config(text="Enable animation")

    def _moving_average(self, values: List[float], window: int = 20) -> List[float]:
        if len(values) < window:
            return values
        
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(values, kernel, mode='valid')
        return smoothed.tolist()

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('Episode', color='white')
        self.ax.set_ylabel('Return', color='white')
        self.ax.set_title('Training Returns', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('#2b2b2b')
        self.ax.spines['right'].set_color('#2b2b2b')
        self.ax.grid(True, alpha=0.2, color='white')

        z_order_base = 1
        for method_config in self.methods:
            method_name = method_config.name
            if method_name in self.returns_by_method:
                returns = self.returns_by_method[method_name]
                if len(returns) > 0:
                    episodes = list(range(1, len(returns) + 1))
                    
                    # Plot raw returns (light color, background)
                    self.ax.plot(episodes, returns, color=method_config.color_light, 
                                alpha=0.4, linewidth=1, zorder=z_order_base)
                    
                    # Plot moving average (bold color, foreground)
                    if len(returns) >= 20:
                        smoothed = self._moving_average(returns, window=20)
                        smooth_episodes = list(range(10, 10 + len(smoothed)))
                        self.ax.plot(smooth_episodes, smoothed, color=method_config.color_bold,
                                   linewidth=2, label=method_name, zorder=z_order_base + 1)
                    else:
                        self.ax.plot(episodes, returns, color=method_config.color_bold,
                                   linewidth=2, label=method_name, zorder=z_order_base + 1)
                    
                    z_order_base += 2

        if self.returns_by_method:
            self.ax.legend(loc='lower left', facecolor='#2b2b2b', edgecolor='white', 
                          labelcolor='white', framealpha=0.8)

        self.canvas.draw()

    def _schedule_queue_poll(self):
        try:
            while True:
                msg_type, data = self._queue.get_nowait()
                
                if msg_type == "return":
                    method_name, episode_return = data
                    if method_name not in self.returns_by_method:
                        self.returns_by_method[method_name] = []
                    self.returns_by_method[method_name].append(episode_return)
                    self._update_plot()
                
                elif msg_type == "render":
                    img_array = data
                    if self._animation_enabled and img_array is not None:
                        img = Image.fromarray(img_array)
                        img = img.resize((600, 400), Image.Resampling.LANCZOS)
                        self._render_image = ImageTk.PhotoImage(img)
                        self.render_canvas.delete("all")
                        self.render_canvas.create_image(300, 200, image=self._render_image)
                
                elif msg_type == "done":
                    method_name = data
                    if method_name in self._workers:
                        del self._workers[method_name]
                    
                    if len(self._workers) == 0:
                        self.train_button.config(state="normal")
                        self.cancel_button.config(state="disabled")
        
        except queue.Empty:
            pass
        
        self.master.after(50, self._schedule_queue_poll)

    def _create_policy(self, method_name: str, state_dim: int, action_dim: int):
        if method_name == "Rainbow":
            return RainbowPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                alpha=self.alpha_var.get(),
                gamma=self.gamma_var.get(),
                eps_start=1.0,
                eps_end=0.01,
                eps_decay=0.001,
                hidden_layers=self.hidden_layers_var.get(),
                hidden_units=self.hidden_units_var.get(),
                batch_size=self.batch_size_var.get(),
                replay_size=self.rainbow_replay_size_var.get(),
                activation=self.activation_var.get(),
                warmup_steps=1000,
                grad_clip=self.grad_clip_var.get(),
                target_update=self.rainbow_target_update_var.get(),
                priority_alpha=0.6,
                beta_start=0.4,
                beta_end=1.0,
                beta_steps=100000,
                atoms=self.rainbow_atoms_var.get(),
                vmin=self.rainbow_vmin_var.get(),
                vmax=self.rainbow_vmax_var.get(),
                noisy_sigma=self.rainbow_noisy_sigma_var.get(),
                n_step=3,
            )
        elif method_name == "A2C":
            return A2CPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                alpha=self.alpha_var.get(),
                gamma=self.gamma_var.get(),
                hidden_layers=self.hidden_layers_var.get(),
                hidden_units=self.hidden_units_var.get(),
                activation=self.activation_var.get(),
                grad_clip=self.grad_clip_var.get(),
                value_coef=self.a2c_value_coef_var.get(),
                entropy_coef=self.a2c_entropy_coef_var.get(),
                n_steps=self.a2c_n_steps_var.get(),
            )
        elif method_name == "TRPO":
            return TRPOPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                gamma=self.gamma_var.get(),
                hidden_layers=self.hidden_layers_var.get(),
                hidden_units=self.hidden_units_var.get(),
                activation=self.activation_var.get(),
                max_kl=self.trpo_max_kl_var.get(),
                damping=self.trpo_damping_var.get(),
                value_lr=self.alpha_var.get(),
                n_steps=self.n_steps_var.get(),
                gae_lambda=self.trpo_gae_lambda_var.get(),
            )
        elif method_name == "PPO":
            return PPOPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                alpha=self.alpha_var.get(),
                gamma=self.gamma_var.get(),
                hidden_layers=self.hidden_layers_var.get(),
                hidden_units=self.hidden_units_var.get(),
                activation=self.activation_var.get(),
                grad_clip=self.grad_clip_var.get(),
                clip_epsilon=self.ppo_clip_epsilon_var.get(),
                value_coef=self.ppo_value_coef_var.get(),
                entropy_coef=self.ppo_entropy_coef_var.get(),
                n_steps=self.n_steps_var.get(),
                n_epochs=self.ppo_n_epochs_var.get(),
                batch_size=self.batch_size_var.get(),
                gae_lambda=self.ppo_gae_lambda_var.get(),
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def _worker_thread(self, method_name: str, episodes: int, stop_event: threading.Event):
        seed_str = self.seed_var.get()
        seed = int(seed_str) if seed_str else None
        
        # Create separate environment for this worker
        env = LunarLanderCEnv(
            seed=seed,
            render_mode="rgb_array" if method_name == self._render_method else None,
            continuous=self.continuous_var.get(),
            gravity=self.gravity_var.get(),
            enable_wind=self.enable_wind_var.get(),
            wind_power=self.wind_power_var.get(),
            turbulence_power=self.turbulence_power_var.get(),
        )
        
        obs_shape = env.observation_space.shape
        if obs_shape is None:
            raise ValueError("Observation space must have a shape.")
        state_dim = int(obs_shape[0])
        action_dim = env.action_dim
        
        policy = self._create_policy(method_name, state_dim, action_dim)
        agent = Agent(env, policy)
        
        last_render_time = time.time()
        render_interval = self.render_interval_var.get()
        
        def render_callback():
            nonlocal last_render_time
            if method_name == self._render_method:
                current_time = time.time()
                if current_time - last_render_time >= render_interval:
                    img = env.render()
                    if img is not None:
                        self._queue.put(("render", img))
                    last_render_time = current_time
        
        for episode in range(1, episodes + 1):
            if stop_event.is_set():
                break
            
            episode_return, steps = agent.run_episode(
                max_steps=self.max_steps_var.get(),
                render_callback=render_callback,
                step_delay=self.step_delay_var.get(),
                stop_event=stop_event,
            )
            
            self._queue.put(("return", (method_name, episode_return)))
        
        env.close()
        self._queue.put(("done", method_name))

    def _on_train(self):
        self.train_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        
        episodes = self.episodes_var.get()
        
        if self.compare_methods.get():
            methods_to_run = [m.name for m in self.methods]
        else:
            methods_to_run = [self.method_var.get()]
        
        # Set render method to the last selected method
        self._render_method = methods_to_run[-1]
        
        for method_name in methods_to_run:
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._worker_thread,
                args=(method_name, episodes, stop_event),
                daemon=True,
            )
            self._workers[method_name] = {
                "thread": thread,
                "stop_event": stop_event,
            }
            thread.start()

    def _on_cancel(self):
        for worker_data in self._workers.values():
            worker_data["stop_event"].set()

    def _on_reset(self):
        self._on_cancel()
        time.sleep(0.5)
        
        self.returns_by_method.clear()
        self._workers.clear()
        self._update_plot()
        
        self.render_canvas.delete("all")
        self._render_image = None
        
        self.train_button.config(state="normal")
        self.cancel_button.config(state="disabled")

    def _on_save_plot(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile="lunar_lander_c_plot.png",
        )
        if filename:
            self.fig.savefig(filename, facecolor='#2b2b2b', edgecolor='none')
