"""
Pusher GUI with Tkinter
Provides interactive interface for training and visualizing RL algorithms
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import cv2

from pusher_logic import Agent, DDPG, TD3, SAC, PusherEnvironment


class PusherGUI:
    """GUI for Pusher RL training and visualization"""

    def __init__(self, root, environment, agent, policies):
        self.root = root
        self.environment = environment
        self.agent = agent
        self.policies = policies

        self.root.title("Pusher RL Training")
        self.root.geometry("1400x900")

        # Dark theme colors
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.entry_bg = "#3c3f41"
        self.button_bg = "#4a4a4a"

        self.root.configure(bg=self.bg_color)

        # Training control
        self.training = False
        self.training_threads = {}
        self.compare_mode = False
        self.paused_methods = set()

        # Plot stats per run label
        self.method_stats = {}

        # Animation
        self.show_animation = tk.BooleanVar(value=False)
        self.animation_method_var = tk.StringVar(value=list(self.policies.keys())[0])
        self.animation_canvas = None
        self.animation_image_id = None

        # Parameters
        self.param_vars = {}
        self.env_param_vars = {}

        # Live grid
        self.grid_rows = {}

        self._create_widgets()
        self._setup_plot()

    def _get_color(self, method_name):
        colors = {
            "DDPG": "#f7b731",
            "TD3": "#4ecdc4",
            "SAC": "#5f27cd"
        }
        return colors.get(method_name, "#95afc0")

    def _create_widgets(self):
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # === TOP SECTION (Animation) ===
        top_frame = tk.Frame(main_frame, bg=self.bg_color)
        top_frame.grid(row=0, column=0, sticky="nsew", pady=5)

        top_frame.grid_columnconfigure(0, weight=2)
        top_frame.grid_columnconfigure(1, weight=1)
        top_frame.grid_rowconfigure(0, weight=1)

        anim_frame = tk.LabelFrame(top_frame, text="Animation", bg=self.bg_color, fg=self.fg_color)
        anim_frame.grid(row=0, column=0, sticky="nsew", padx=5)

        self.animation_canvas = tk.Canvas(anim_frame, width=520, height=300,
                                          bg=self.bg_color, highlightthickness=0)
        self.animation_canvas.pack(pady=8, padx=8, fill=tk.BOTH, expand=True)

        anim_controls = tk.Frame(anim_frame, bg=self.bg_color)
        anim_controls.pack(fill=tk.X, padx=8, pady=5)

        tk.Checkbutton(anim_controls, text="Enable Animation",
                       variable=self.show_animation, bg=self.bg_color, fg=self.fg_color,
                       selectcolor=self.entry_bg, command=self._toggle_animation).pack(side=tk.LEFT)

        tk.Label(anim_controls, text="Animation Method:", bg=self.bg_color,
                 fg=self.fg_color).pack(side=tk.LEFT, padx=8)
        ttk.Combobox(anim_controls, textvariable=self.animation_method_var,
                     values=list(self.policies.keys()), state="readonly",
                     width=10).pack(side=tk.LEFT)

        env_frame = tk.LabelFrame(top_frame, text="Environment Settings", bg=self.bg_color, fg=self.fg_color)
        env_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        env_grid = tk.Frame(env_frame, bg=self.bg_color)
        env_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        env_params = [
            ("reward_near_weight", "0.5"),
            ("reward_dist_weight", "1.0"),
            ("reward_control_weight", "0.1")
        ]
        for i, (key, default) in enumerate(env_params):
            tk.Label(env_grid, text=key + ":", bg=self.bg_color, fg=self.fg_color).grid(
                row=i, column=0, sticky="w", padx=2, pady=2
            )
            var = tk.StringVar(value=default)
            self.env_param_vars[key] = var
            tk.Entry(env_grid, textvariable=var, bg=self.entry_bg, fg=self.fg_color,
                     width=12).grid(row=i, column=1, padx=2, pady=2)

        tk.Button(env_frame, text="Update Environment", command=self._update_environment,
                  bg=self.button_bg, fg=self.fg_color).pack(fill=tk.X, padx=5, pady=5)

        # === MIDDLE SECTION (Controls) ===
        middle_frame = tk.Frame(main_frame, bg=self.bg_color)
        middle_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        middle_frame.grid_rowconfigure(0, weight=2)
        middle_frame.grid_rowconfigure(1, weight=1)
        middle_frame.grid_columnconfigure(0, weight=1)

        upper_controls = tk.Frame(middle_frame, bg=self.bg_color)
        upper_controls.grid(row=0, column=0, sticky="nsew", padx=5)

        lower_grid = tk.Frame(middle_frame, bg=self.bg_color)
        lower_grid.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Upper controls
        method_frame = tk.LabelFrame(upper_controls, text="Method Selection", bg=self.bg_color, fg=self.fg_color)
        method_frame.pack(fill=tk.X, pady=5)

        method_select_row = tk.Frame(method_frame, bg=self.bg_color)
        method_select_row.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(method_select_row, text="Algorithm:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value=list(self.policies.keys())[0])
        ttk.Combobox(method_select_row, textvariable=self.method_var,
                     values=list(self.policies.keys()), state="readonly",
                     width=10).pack(side=tk.LEFT, padx=5)

        self.compare_var = tk.BooleanVar(value=False)
        tk.Checkbutton(method_select_row, text="Compare Mode",
                       variable=self.compare_var, bg=self.bg_color, fg=self.fg_color,
                       selectcolor=self.entry_bg, command=self._toggle_compare_mode).pack(side=tk.LEFT, padx=10)

        common_frame = tk.LabelFrame(upper_controls, text="Common Parameters", bg=self.bg_color, fg=self.fg_color)
        common_frame.pack(fill=tk.X, pady=5)

        params_grid = tk.Frame(common_frame, bg=self.bg_color)
        params_grid.pack(fill=tk.X, padx=5, pady=5)

        common_params = [
            ("Episodes", "episodes", "200"),
            ("Learning Rate", "lr", "0.0003"),
            ("Gamma", "gamma", "0.99"),
            ("Hidden Dims", "hidden_dims", "256,256"),
            ("Network Type", "network_type", "mlp"),
            ("Activation", "activation", "relu")
        ]

        for idx, (label, key, default) in enumerate(common_params):
            row = idx // 3
            col = idx % 3
            tk.Label(params_grid, text=label + ":", bg=self.bg_color, fg=self.fg_color).grid(
                row=row * 2, column=col, sticky="w", padx=2, pady=2
            )
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            tk.Entry(params_grid, textvariable=var, bg=self.entry_bg, fg=self.fg_color,
                     width=12).grid(row=row * 2 + 1, column=col, padx=2, pady=2)

        notebook_frame = tk.LabelFrame(upper_controls, text="Method Parameters", bg=self.bg_color, fg=self.fg_color)
        notebook_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._create_method_tabs()

        # Live grid
        grid_title = tk.LabelFrame(lower_grid, text="Live Grid", bg=self.bg_color, fg=self.fg_color)
        grid_title.pack(fill=tk.BOTH, expand=True)

        self._create_live_grid(grid_title)

        # === BOTTOM SECTION (Plot) ===
        bottom_frame = tk.Frame(main_frame, bg=self.bg_color)
        bottom_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        bottom_frame.grid_rowconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(0, weight=1)

        button_frame = tk.Frame(bottom_frame, bg=self.bg_color)
        button_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        tk.Button(button_frame, text="Reset", command=self._reset_training,
                  bg=self.button_bg, fg=self.fg_color, width=12).pack(side=tk.LEFT, padx=4)
        tk.Button(button_frame, text="Train", command=self._start_training,
                  bg=self.button_bg, fg=self.fg_color, width=12).pack(side=tk.LEFT, padx=4)
        tk.Button(button_frame, text="Run", command=self._run_episode,
                  bg=self.button_bg, fg=self.fg_color, width=12).pack(side=tk.LEFT, padx=4)
        tk.Button(button_frame, text="Cancel", command=self._stop_training,
                  bg=self.button_bg, fg=self.fg_color, width=12).pack(side=tk.LEFT, padx=4)
        tk.Button(button_frame, text="Save Plot", command=self._save_plot,
                  bg=self.button_bg, fg=self.fg_color, width=12).pack(side=tk.LEFT, padx=4)

        plot_frame = tk.LabelFrame(bottom_frame, text="Reward Plot", bg=self.bg_color, fg=self.fg_color)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.plot_canvas_frame = tk.Frame(plot_frame, bg=self.bg_color)
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Status
        status_frame = tk.LabelFrame(bottom_frame, text="Status", bg=self.bg_color, fg=self.fg_color)
        status_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        self.status_text = tk.Text(status_frame, height=4, bg=self.entry_bg,
                                   fg=self.fg_color, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_method_tabs(self):
        for method in self.policies.keys():
            tab = tk.Frame(self.notebook, bg=self.bg_color)
            self.notebook.add(tab, text=method)

            params = []
            if method == "DDPG":
                params = [
                    ("Tau", "tau", "0.005"),
                    ("Buffer Size", "buffer_size", "100000"),
                    ("Batch Size", "batch_size", "64"),
                    ("Exploration Noise", "exploration_noise", "0.1"),
                    ("Actor Activation", "actor_activation", "relu"),
                    ("Critic Activation", "critic_activation", "relu")
                ]
            elif method == "TD3":
                params = [
                    ("Tau", "tau", "0.005"),
                    ("Policy Noise", "policy_noise", "0.2"),
                    ("Noise Clip", "noise_clip", "0.5"),
                    ("Policy Delay", "policy_delay", "2"),
                    ("Buffer Size", "buffer_size", "100000"),
                    ("Batch Size", "batch_size", "64"),
                    ("Exploration Noise", "exploration_noise", "0.1"),
                    ("Actor Activation", "actor_activation", "relu"),
                    ("Critic Activation", "critic_activation", "relu")
                ]
            elif method == "SAC":
                params = [
                    ("Tau", "tau", "0.005"),
                    ("Alpha", "alpha", "0.2"),
                    ("Buffer Size", "buffer_size", "100000"),
                    ("Batch Size", "batch_size", "64"),
                    ("Train Freq", "train_freq", "4"),
                    ("Gradient Steps", "gradient_steps", "1"),
                    ("Actor Activation", "actor_activation", "relu"),
                    ("Critic Activation", "critic_activation", "relu"),
                    ("Alpha From", "alpha_from", "0.2"),
                    ("Alpha To", "alpha_to", "0.2"),
                    ("Alpha Step", "alpha_step", "0.0"),
                    ("Buffer From", "buffer_from", "100000"),
                    ("Buffer To", "buffer_to", "100000"),
                    ("Buffer Step", "buffer_step", "0"),
                    ("Gamma From", "gamma_from", "0.99"),
                    ("Gamma To", "gamma_to", "0.99"),
                    ("Gamma Step", "gamma_step", "0.0")
                ]

            grid = tk.Frame(tab, bg=self.bg_color)
            grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            for idx, (label, key, default) in enumerate(params):
                row = idx // 3
                col = idx % 3
                tk.Label(grid, text=label + ":", bg=self.bg_color, fg=self.fg_color).grid(
                    row=row * 2, column=col, sticky="w", padx=2, pady=2
                )
                param_key = f"{method}_{key}"
                if param_key not in self.param_vars:
                    self.param_vars[param_key] = tk.StringVar(value=default)
                tk.Entry(grid, textvariable=self.param_vars[param_key], bg=self.entry_bg,
                         fg=self.fg_color, width=12).grid(row=row * 2 + 1, column=col, padx=2, pady=2)

    def _create_live_grid(self, parent):
        headers = [
            "Active", "Method", "Value", "Episode/Total", "Step",
            "Moving Avg", "Reward", "Duration", "Pause", "Animation"
        ]
        header_frame = tk.Frame(parent, bg=self.bg_color)
        header_frame.pack(fill=tk.X, padx=5, pady=2)

        for idx, title in enumerate(headers):
            tk.Label(header_frame, text=title, bg=self.bg_color, fg=self.fg_color, width=12).grid(
                row=0, column=idx, padx=2
            )

        rows_frame = tk.Frame(parent, bg=self.bg_color)
        rows_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        for row_idx, method in enumerate(self.policies.keys()):
            row = {}
            row["active_var"] = tk.BooleanVar(value=False)
            row["value_var"] = tk.StringVar(value="-")
            row["episode_var"] = tk.StringVar(value="0/0")
            row["step_var"] = tk.StringVar(value="0")
            row["avg_var"] = tk.StringVar(value="0.0")
            row["reward_var"] = tk.StringVar(value="0.0")
            row["duration_var"] = tk.StringVar(value="0.0s")

            tk.Checkbutton(rows_frame, variable=row["active_var"], bg=self.bg_color,
                           fg=self.fg_color, selectcolor=self.entry_bg).grid(row=row_idx, column=0)
            tk.Label(rows_frame, text=method, bg=self.bg_color, fg=self.fg_color, width=12).grid(
                row=row_idx, column=1, padx=2
            )
            tk.Label(rows_frame, textvariable=row["value_var"], bg=self.bg_color,
                     fg=self.fg_color, width=12).grid(row=row_idx, column=2, padx=2)
            tk.Label(rows_frame, textvariable=row["episode_var"], bg=self.bg_color,
                     fg=self.fg_color, width=12).grid(row=row_idx, column=3, padx=2)
            tk.Label(rows_frame, textvariable=row["step_var"], bg=self.bg_color,
                     fg=self.fg_color, width=10).grid(row=row_idx, column=4, padx=2)
            tk.Label(rows_frame, textvariable=row["avg_var"], bg=self.bg_color,
                     fg=self.fg_color, width=12).grid(row=row_idx, column=5, padx=2)
            tk.Label(rows_frame, textvariable=row["reward_var"], bg=self.bg_color,
                     fg=self.fg_color, width=10).grid(row=row_idx, column=6, padx=2)
            tk.Label(rows_frame, textvariable=row["duration_var"], bg=self.bg_color,
                     fg=self.fg_color, width=12).grid(row=row_idx, column=7, padx=2)

            pause_btn = tk.Button(rows_frame, text="Pause", bg=self.button_bg, fg=self.fg_color,
                                  width=10, command=lambda m=method: self._toggle_pause(m))
            pause_btn.grid(row=row_idx, column=8, padx=2)
            row["pause_btn"] = pause_btn

            tk.Radiobutton(rows_frame, variable=self.animation_method_var, value=method,
                           bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).grid(
                row=row_idx, column=9, padx=2
            )

            self.grid_rows[method] = row

    def _setup_plot(self):
        self.fig = Figure(figsize=(10, 4), facecolor="#2b2b2b")
        self.ax = self.fig.add_subplot(111, facecolor="#2b2b2b")
        self.ax.set_xlabel("Episode", color="white")
        self.ax.set_ylabel("Reward", color="white")
        self.ax.set_title("Training Progress", color="white")
        self.ax.tick_params(colors="white")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _toggle_animation(self):
        if self.show_animation.get():
            self._log_status("Animation enabled (restart training to apply).")
        else:
            if self.animation_canvas:
                self.animation_canvas.delete("all")
            self._log_status("Animation disabled")

    def _update_animation_frame(self, frame):
        if frame is None or not self.show_animation.get():
            return

        try:
            frame_rgb = frame
            canvas_width = 520
            canvas_height = 300
            h, w = frame_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)

            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            img = Image.fromarray(frame_resized.astype("uint8"))
            photo = ImageTk.PhotoImage(image=img)

            def update_canvas():
                if self.animation_image_id:
                    self.animation_canvas.delete(self.animation_image_id)
                x = (canvas_width - new_w) // 2
                y = (canvas_height - new_h) // 2
                self.animation_image_id = self.animation_canvas.create_image(
                    x, y, anchor=tk.NW, image=photo
                )
                self.animation_canvas.image = photo

            self.root.after(0, update_canvas)
        except Exception as exc:
            self._log_status(f"Animation error: {str(exc)}")

    def _update_environment(self):
        if self.training:
            messagebox.showwarning("Warning", "Stop training before updating environment.")
            return
        try:
            self.environment.reward_near_weight = float(self.env_param_vars["reward_near_weight"].get())
            self.environment.reward_dist_weight = float(self.env_param_vars["reward_dist_weight"].get())
            self.environment.reward_control_weight = float(self.env_param_vars["reward_control_weight"].get())
            self.environment.reset_env()
            self._log_status("Environment updated")
        except ValueError as exc:
            messagebox.showerror("Error", f"Invalid parameter value: {str(exc)}")

    def _toggle_compare_mode(self):
        self.compare_mode = self.compare_var.get()

    def _toggle_pause(self, method_name):
        if method_name in self.paused_methods:
            self.paused_methods.remove(method_name)
            self.grid_rows[method_name]["pause_btn"].config(text="Pause")
        else:
            self.paused_methods.add(method_name)
            self.grid_rows[method_name]["pause_btn"].config(text="Resume")

    def _parse_range(self, prefix, is_int=False):
        from_var = self.param_vars.get(f"SAC_{prefix}_from")
        to_var = self.param_vars.get(f"SAC_{prefix}_to")
        step_var = self.param_vars.get(f"SAC_{prefix}_step")
        if not from_var or not to_var or not step_var:
            return None
        try:
            start = float(from_var.get())
            end = float(to_var.get())
            step = float(step_var.get())
        except ValueError:
            return None
        if step == 0 or start == end:
            return None
        if step < 0:
            step = abs(step)
        if start < end:
            values = list(np.arange(start, end + step / 2, step))
        else:
            values = list(np.arange(start, end - step / 2, -step))
        if is_int:
            values = [int(round(v)) for v in values]
        return values

    def _get_common_params(self):
        return {
            "episodes": int(float(self.param_vars["episodes"].get())),
            "lr": float(self.param_vars["lr"].get()),
            "gamma": float(self.param_vars["gamma"].get()),
            "hidden_dims": [int(x.strip()) for x in self.param_vars["hidden_dims"].get().split(",")],
            "network_type": self.param_vars["network_type"].get().strip().lower(),
            "activation": self.param_vars["activation"].get().strip().lower()
        }

    def _get_method_params(self, method_name):
        common = self._get_common_params()
        params = {
            "state_dim": self.environment.get_state_dim(),
            "action_dim": self.environment.get_action_dim(),
            "hidden_dims": common["hidden_dims"],
            "lr": common["lr"],
            "gamma": common["gamma"],
            "network_type": common["network_type"],
            "actor_activation": common["activation"],
            "critic_activation": common["activation"]
        }

        if method_name == "DDPG":
            params.update({
                "tau": float(self.param_vars["DDPG_tau"].get()),
                "buffer_size": int(float(self.param_vars["DDPG_buffer_size"].get())),
                "batch_size": int(float(self.param_vars["DDPG_batch_size"].get())),
                "exploration_noise": float(self.param_vars["DDPG_exploration_noise"].get()),
                "actor_activation": self.param_vars["DDPG_actor_activation"].get(),
                "critic_activation": self.param_vars["DDPG_critic_activation"].get()
            })
        elif method_name == "TD3":
            params.update({
                "tau": float(self.param_vars["TD3_tau"].get()),
                "policy_noise": float(self.param_vars["TD3_policy_noise"].get()),
                "noise_clip": float(self.param_vars["TD3_noise_clip"].get()),
                "policy_delay": int(float(self.param_vars["TD3_policy_delay"].get())),
                "buffer_size": int(float(self.param_vars["TD3_buffer_size"].get())),
                "batch_size": int(float(self.param_vars["TD3_batch_size"].get())),
                "exploration_noise": float(self.param_vars["TD3_exploration_noise"].get()),
                "actor_activation": self.param_vars["TD3_actor_activation"].get(),
                "critic_activation": self.param_vars["TD3_critic_activation"].get()
            })
        elif method_name == "SAC":
            params.update({
                "tau": float(self.param_vars["SAC_tau"].get()),
                "alpha": float(self.param_vars["SAC_alpha"].get()),
                "buffer_size": int(float(self.param_vars["SAC_buffer_size"].get())),
                "batch_size": int(float(self.param_vars["SAC_batch_size"].get())),
                "train_freq": int(float(self.param_vars["SAC_train_freq"].get())),
                "gradient_steps": int(float(self.param_vars["SAC_gradient_steps"].get())),
                "actor_activation": self.param_vars["SAC_actor_activation"].get(),
                "critic_activation": self.param_vars["SAC_critic_activation"].get()
            })

        return params

    def _reset_training(self):
        self._stop_training()
        self.method_stats = {}
        for method in self.grid_rows.values():
            method["value_var"].set("-")
            method["episode_var"].set("0/0")
            method["step_var"].set("0")
            method["avg_var"].set("0.0")
            method["reward_var"].set("0.0")
            method["duration_var"].set("0.0s")
        self.agent.reset_stats()
        self._update_plot()
        self._log_status("Training reset")

    def _start_training(self):
        if self.training:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        self.training = True
        methods = []
        if self.compare_mode:
            methods = [name for name, row in self.grid_rows.items() if row["active_var"].get()]
            if not methods:
                messagebox.showwarning("Warning", "No methods selected in the live grid")
                self.training = False
                return
        else:
            methods = [self.method_var.get()]
            self.grid_rows[methods[0]]["active_var"].set(True)

        for method_name in methods:
            thread = threading.Thread(target=self._train_method, args=(method_name,))
            thread.daemon = True
            self.training_threads[method_name] = thread
            thread.start()

    def _create_environment(self, render_mode=None):
        return PusherEnvironment(
            reward_near_weight=float(self.env_param_vars["reward_near_weight"].get()),
            reward_dist_weight=float(self.env_param_vars["reward_dist_weight"].get()),
            reward_control_weight=float(self.env_param_vars["reward_control_weight"].get()),
            render_mode=render_mode
        )

    def _train_method(self, method_name):
        start_time = time.time()
        try:
            common = self._get_common_params()
            episodes = common["episodes"]
            policy_class = self.policies[method_name]
            base_params = self._get_method_params(method_name)

            range_overrides = [({"label": "default"})]

            if method_name == "SAC":
                alpha_values = self._parse_range("alpha")
                buffer_values = self._parse_range("buffer", is_int=True)
                gamma_values = self._parse_range("gamma")

                if alpha_values:
                    range_overrides = []
                    for val in alpha_values:
                        range_overrides.append({"alpha": float(val), "label": f"alpha={val:.3g}"})
                if buffer_values:
                    new_overrides = []
                    for base in range_overrides:
                        for val in buffer_values:
                            label = f"{base.get('label', 'default')},buffer={val}"
                            new = dict(base)
                            new["buffer_size"] = int(val)
                            new["label"] = label
                            new_overrides.append(new)
                    range_overrides = new_overrides
                if gamma_values:
                    new_overrides = []
                    for base in range_overrides:
                        for val in gamma_values:
                            label = f"{base.get('label', 'default')},gamma={val:.3g}"
                            new = dict(base)
                            new["gamma"] = float(val)
                            new["label"] = label
                            new_overrides.append(new)
                    range_overrides = new_overrides

            if len(range_overrides) > 20:
                range_overrides = range_overrides[:20]
                self._log_status(f"Too many range combinations for {method_name}; limiting to 20.")

            for override in range_overrides:
                if not self.training:
                    break

                label = override.get("label", "default")
                label = label.replace("default,", "").replace("default", "").strip(",") or "default"
                run_label = method_name if label == "default" else f"{method_name} ({label})"
                self.grid_rows[method_name]["value_var"].set(label)

                params = dict(base_params)
                params.update({k: v for k, v in override.items() if k != "label"})

                render_mode = None
                if self.show_animation.get() and self.animation_method_var.get() == method_name:
                    render_mode = "rgb_array"

                env = self._create_environment(render_mode=render_mode)
                agent = Agent(env, policy_class(**params))

                if run_label not in self.method_stats:
                    self.method_stats[run_label] = {
                        "returns": [],
                        "color": self._get_color(method_name)
                    }

                self._log_status(f"Training {run_label} for {episodes} episodes...")

                for episode in range(episodes):
                    if not self.training:
                        break

                    while method_name in self.paused_methods and self.training:
                        time.sleep(0.1)

                    render = self.show_animation.get() and self.animation_method_var.get() == method_name
                    render_callback = self._update_animation_frame if render else None

                    reward, steps = agent.run_episode(render=render, train=True, render_callback=render_callback)

                    returns = self.method_stats[run_label]["returns"]
                    returns.append(reward)

                    avg_reward = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
                    duration = time.time() - start_time

                    self._update_grid(method_name, episode + 1, episodes, steps, avg_reward, reward, duration)

                    if episode < 20 or episode % 5 == 0 or episode == episodes - 1:
                        self.root.after(0, self._update_plot)

                env.close()

            self._log_status(f"{method_name} training completed")

        except Exception as exc:
            self._log_status(f"Error training {method_name}: {str(exc)}")
        finally:
            if method_name in self.training_threads:
                del self.training_threads[method_name]
            if len(self.training_threads) == 0:
                self.training = False

    def _update_grid(self, method_name, episode, episodes, steps, avg_reward, reward, duration):
        def update():
            row = self.grid_rows[method_name]
            row["episode_var"].set(f"{episode}/{episodes}")
            row["step_var"].set(str(steps))
            row["avg_var"].set(f"{avg_reward:.2f}")
            row["reward_var"].set(f"{reward:.2f}")
            row["duration_var"].set(f"{duration:.1f}s")

        if threading.current_thread() == threading.main_thread():
            update()
        else:
            self.root.after(0, update)

    def _run_episode(self):
        if self.training:
            messagebox.showwarning("Warning", "Stop training before running a single episode.")
            return

        method_name = self.method_var.get()
        params = self._get_method_params(method_name)
        policy_class = self.policies[method_name]
        self.agent.set_policy(policy_class(**params))

        thread = threading.Thread(target=self._run_episode_thread)
        thread.daemon = True
        thread.start()

    def _run_episode_thread(self):
        try:
            render = self.show_animation.get()
            render_callback = self._update_animation_frame if render else None
            reward, steps = self.agent.run_episode(render=render, train=False, render_callback=render_callback)
            self._log_status(f"Episode reward: {reward:.2f}, steps: {steps}")
        except Exception as exc:
            self._log_status(f"Error running episode: {str(exc)}")

    def _stop_training(self):
        self.training = False
        self.training_threads.clear()
        self._log_status("Training stopped")

    def _update_plot(self):
        self.ax.clear()

        for run_label, stats in self.method_stats.items():
            returns = stats["returns"]
            if not returns:
                continue
            episodes = list(range(1, len(returns) + 1))
            color = stats["color"]

            self.ax.plot(episodes, returns, color=color, alpha=0.3,
                         linewidth=0.6, zorder=1, label=f"{run_label} (raw)")

            if len(returns) >= 10:
                window = min(20, len(returns))
                moving_avg = np.convolve(returns, np.ones(window) / window, mode="valid")
                avg_episodes = list(range(window, len(returns) + 1))
                self.ax.plot(avg_episodes, moving_avg, color=color,
                             linewidth=2, zorder=2, label=f"{run_label} (avg)")

        self.ax.set_xlabel("Episode", color="white")
        self.ax.set_ylabel("Reward", color="white")
        self.ax.set_title("Training Progress", color="white")
        self.ax.tick_params(colors="white")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")
        self.ax.legend(loc="lower left", facecolor="#2b2b2b", edgecolor="white",
                       labelcolor="white", fontsize=7)
        self.ax.grid(True, alpha=0.2, color="white")

        self.canvas.draw()

    def _save_plot(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        method_name = self.method_var.get()
        value_label = self.grid_rows[method_name]["value_var"].get().replace(" ", "_")
        value_part = "" if value_label in ("-", "default") else f"_{value_label}"
        filename = f"pusher_plot_{method_name}{value_part}_{timestamp}.png"
        self.fig.savefig(filename, facecolor="#2b2b2b", edgecolor="white")
        self._log_status(f"Plot saved to {filename}")
        messagebox.showinfo("Success", f"Plot saved to {filename}")

    def _log_status(self, message):
        def update():
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.see(tk.END)

        if threading.current_thread() == threading.main_thread():
            update()
        else:
            self.root.after(0, update)

    def run(self):
        self.root.mainloop()
