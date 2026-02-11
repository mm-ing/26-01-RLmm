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
        self.paused_rows = set()

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
        self.grid_header_frame = None
        self.grid_rows_frame = None

        self._create_widgets()
        self._setup_plot()

    def _get_color(self, method_name):
        colors = {
            "DDPG": "#f7b731",
            "TD3": "#4ecdc4",
            "SAC": "#5f27cd"
        }
        return colors.get(method_name, "#95afc0")

    def _method_order(self):
        preferred = ["DDPG", "TD3", "SAC"]
        return [name for name in preferred if name in self.policies]

    def _create_widgets(self):
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=6)
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
        middle_frame.grid_rowconfigure(0, weight=6, minsize=320)
        middle_frame.grid_rowconfigure(1, weight=2, minsize=140)
        middle_frame.grid_columnconfigure(0, weight=1)

        upper_controls = tk.Frame(middle_frame, bg=self.bg_color)
        upper_controls.grid(row=0, column=0, sticky="nsew", padx=5)

        lower_grid = tk.Frame(middle_frame, bg=self.bg_color)
        lower_grid.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        lower_grid.grid_propagate(True)

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
            label_col = idx * 2
            entry_col = idx * 2 + 1
            tk.Label(params_grid, text=label + ":", bg=self.bg_color, fg=self.fg_color).grid(
                row=0, column=label_col, sticky="w", padx=2, pady=2
            )
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            tk.Entry(params_grid, textvariable=var, bg=self.entry_bg, fg=self.fg_color,
                     width=12).grid(row=0, column=entry_col, padx=2, pady=6,
                                    ipady=6)

        notebook_frame = tk.LabelFrame(upper_controls, text="Method Parameters", bg=self.bg_color, fg=self.fg_color)
        notebook_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        notebook_frame.configure(height=420)
        notebook_frame.pack_propagate(False)

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

        self.status_text = None

    def _create_method_tabs(self):
        for method in self.policies.keys():
            tab = tk.Frame(self.notebook, bg=self.bg_color)
            self.notebook.add(tab, text=method)

            tab_canvas = tk.Canvas(tab, bg=self.bg_color, highlightthickness=0)
            tab_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

            tab_scrollbar = tk.Scrollbar(tab, orient="vertical", command=tab_canvas.yview)
            tab_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            tab_canvas.configure(yscrollcommand=tab_scrollbar.set)

            scroll_frame = tk.Frame(tab_canvas, bg=self.bg_color)
            tab_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

            scroll_frame.bind(
                "<Configure>",
                lambda e, canvas=tab_canvas: canvas.configure(scrollregion=canvas.bbox("all"))
            )

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

            grid = tk.Frame(scroll_frame, bg=self.bg_color)
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
                         fg=self.fg_color, width=12).grid(row=row * 2 + 1, column=col, padx=2, pady=10,
                                                         ipady=10)

    def _create_live_grid(self, parent):
        headers = [
            "Active", "Method", "Param", "Value", "Episode/Total", "Step",
            "Reward", "Moving Avg", "Duration", "Pause", "Animation"
        ]
        col_mins = [60, 120, 110, 110, 120, 70, 80, 110, 110, 80, 100]
        scrollbar_width = 16

        grid_container = tk.Frame(parent, bg=self.bg_color)
        grid_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        grid_container.grid_rowconfigure(1, weight=1)
        grid_container.grid_columnconfigure(0, weight=1)

        self.grid_header_frame = tk.Frame(grid_container, bg=self.bg_color)
        self.grid_header_frame.grid(row=0, column=0, sticky="nsew")

        for col in range(len(headers)):
            min_size = col_mins[col] if col < len(col_mins) else 80
            self.grid_header_frame.grid_columnconfigure(col, weight=1, uniform="grid", minsize=min_size)
        self.grid_header_frame.grid_columnconfigure(len(headers), weight=0, minsize=scrollbar_width)

        for idx, title in enumerate(headers):
            tk.Label(self.grid_header_frame, text=title, bg=self.bg_color,
                     fg=self.fg_color).grid(row=0, column=idx, padx=2, pady=1, sticky="nsew")
        tk.Frame(self.grid_header_frame, bg=self.bg_color, width=scrollbar_width).grid(
            row=0, column=len(headers), sticky="nsew"
        )

        grid_body = tk.Frame(grid_container, bg=self.bg_color)
        grid_body.grid(row=1, column=0, sticky="nsew")
        grid_body.grid_columnconfigure(0, weight=1)
        grid_body.grid_rowconfigure(0, weight=1)

        self.grid_canvas = tk.Canvas(grid_body, bg=self.bg_color, highlightthickness=0)
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.Scrollbar(grid_body, orient="vertical", command=self.grid_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.grid_canvas.configure(yscrollcommand=scrollbar.set)

        self.grid_rows_frame = tk.Frame(self.grid_canvas, bg=self.bg_color)
        self.grid_rows_window = self.grid_canvas.create_window(
            (0, 0), window=self.grid_rows_frame, anchor="nw"
        )

        self.grid_rows_frame.bind(
            "<Configure>",
            lambda e: self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all"))
        )

        self.grid_canvas.bind(
            "<Configure>",
            lambda e: self._sync_grid_width(e.width)
        )

        for col in range(len(headers)):
            min_size = col_mins[col] if col < len(col_mins) else 80
            self.grid_rows_frame.grid_columnconfigure(col, weight=1, uniform="grid", minsize=min_size)

        self._build_live_grid_rows(self._base_grid_defs())

    def _sync_grid_width(self, canvas_width):
        if canvas_width <= 0:
            return
        self.grid_canvas.itemconfigure(self.grid_rows_window, width=canvas_width)
        self.grid_header_frame.configure(width=canvas_width)

    def _base_grid_defs(self):
        base_methods = self._method_order()
        return [
            {
                "row_key": method,
                "method": method,
                "param": "",
                "value": "",
                "is_base": True,
                "active": idx == 0
            }
            for idx, method in enumerate(base_methods)
        ]

    def _clear_live_grid_rows(self):
        for widget in self.grid_rows_frame.winfo_children():
            widget.destroy()
        self.grid_rows = {}

    def _build_live_grid_rows(self, row_defs):
        self._clear_live_grid_rows()

        for row_idx, row_def in enumerate(row_defs):
            row_key = row_def["row_key"]
            method = row_def["method"]
            param_name = row_def.get("param", "")
            param_value = row_def.get("value", "")
            is_base = row_def.get("is_base", False)
            is_active = row_def.get("active", False)

            row = {
                "row_key": row_key,
                "method": method,
                "param_var": tk.StringVar(value=param_name or "-"),
                "value_var": tk.StringVar(value=param_value or "-"),
                "episode_var": tk.StringVar(value="0/0"),
                "step_var": tk.StringVar(value="0"),
                "reward_var": tk.StringVar(value="0.0"),
                "avg_var": tk.StringVar(value="0.0"),
                "duration_var": tk.StringVar(value="00:00:00"),
                "is_base": is_base
            }

            active_var = tk.BooleanVar(value=is_active if is_base else True)
            row["active_var"] = active_var
            active_btn = tk.Checkbutton(self.grid_rows_frame, variable=active_var, bg=self.bg_color,
                                        fg=self.fg_color, selectcolor=self.entry_bg)
            active_btn.grid(row=row_idx, column=0, padx=2, pady=1, sticky="nsew")
            if not is_base:
                active_btn.configure(state=tk.DISABLED)

            tk.Label(self.grid_rows_frame, text=method, bg=self.bg_color, fg=self.fg_color).grid(
                row=row_idx, column=1, padx=2, pady=1, sticky="nsew"
            )
            tk.Label(self.grid_rows_frame, textvariable=row["param_var"], bg=self.bg_color,
                     fg=self.fg_color).grid(row=row_idx, column=2, padx=2, pady=1, sticky="nsew")
            tk.Label(self.grid_rows_frame, textvariable=row["value_var"], bg=self.bg_color,
                     fg=self.fg_color).grid(row=row_idx, column=3, padx=2, pady=1, sticky="nsew")
            tk.Label(self.grid_rows_frame, textvariable=row["episode_var"], bg=self.bg_color,
                     fg=self.fg_color).grid(row=row_idx, column=4, padx=2, pady=1, sticky="nsew")
            tk.Label(self.grid_rows_frame, textvariable=row["step_var"], bg=self.bg_color,
                     fg=self.fg_color).grid(row=row_idx, column=5, padx=2, pady=1, sticky="nsew")
            tk.Label(self.grid_rows_frame, textvariable=row["reward_var"], bg=self.bg_color,
                     fg=self.fg_color).grid(row=row_idx, column=6, padx=2, pady=1, sticky="nsew")
            tk.Label(self.grid_rows_frame, textvariable=row["avg_var"], bg=self.bg_color,
                     fg=self.fg_color).grid(row=row_idx, column=7, padx=2, pady=1, sticky="nsew")
            tk.Label(self.grid_rows_frame, textvariable=row["duration_var"], bg=self.bg_color,
                     fg=self.fg_color).grid(row=row_idx, column=8, padx=2, pady=1, sticky="nsew")

            pause_btn = tk.Button(self.grid_rows_frame, text="Pause", bg=self.button_bg, fg=self.fg_color,
                                  command=lambda key=row_key: self._toggle_pause(key))
            pause_btn.grid(row=row_idx, column=9, padx=2, pady=1, sticky="nsew")
            row["pause_btn"] = pause_btn

            anim_btn = tk.Radiobutton(self.grid_rows_frame, variable=self.animation_method_var, value=method,
                                      bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg,
                                      command=lambda m=method: self._activate_animation(m))
            anim_btn.grid(row=row_idx, column=10, padx=2, pady=1, sticky="nsew")

            self.grid_rows[row_key] = row

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
            self._set_env_render_mode("rgb_array")
            self._log_status("Animation enabled (restart training to apply).")
        else:
            self._set_env_render_mode(None)
            if self.animation_canvas:
                self.animation_canvas.delete("all")
            self._log_status("Animation disabled")

    def _activate_animation(self, method_name):
        self.animation_method_var.set(method_name)
        if not self.show_animation.get():
            self.show_animation.set(True)
        self._toggle_animation()

    def _set_env_render_mode(self, render_mode):
        self.environment.render_mode = render_mode
        self.environment.reset_env()

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

    def _get_selected_methods(self):
        base_rows = [row for row in self.grid_rows.values() if row.get("is_base")]
        selected = [row["method"] for row in base_rows if row["active_var"].get()]
        if selected:
            return selected
        ordered = self._method_order()
        return [ordered[0]] if ordered else []

    def _toggle_pause(self, row_key):
        if row_key in self.paused_rows:
            self.paused_rows.remove(row_key)
            self.grid_rows[row_key]["pause_btn"].config(text="Pause")
        else:
            self.paused_rows.add(row_key)
            self.grid_rows[row_key]["pause_btn"].config(text="Resume")

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

    def _format_duration(self, seconds):
        total = int(seconds)
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _get_range_overrides(self, method_name):
        if method_name != "SAC":
            return [{"label": "default"}], None

        alpha_values = self._parse_range("alpha")
        buffer_values = self._parse_range("buffer", is_int=True)
        gamma_values = self._parse_range("gamma")

        candidates = [
            ("alpha", alpha_values),
            ("buffer", buffer_values),
            ("gamma", gamma_values)
        ]
        selected = [(name, values) for name, values in candidates if values]

        if len(selected) > 1:
            self._log_status("Multiple parameter ranges set; using the first one only.")

        if not selected:
            return [{"label": "default"}], None

        param_name, values = selected[0]
        overrides = []
        for val in values:
            if param_name == "alpha":
                overrides.append({"alpha": float(val), "label": f"{val}"})
            elif param_name == "buffer":
                overrides.append({"buffer_size": int(val), "label": f"{val}"})
            else:
                overrides.append({"gamma": float(val), "label": f"{val}"})

        return overrides, param_name

    def _build_training_grid_defs(self, selected_methods):
        row_defs = []
        for method in self._method_order():
            row_defs.append({
                "row_key": method,
                "method": method,
                "param": "",
                "value": "",
                "is_base": True,
                "active": method in selected_methods
            })

        for method in selected_methods:
            overrides, param_name = self._get_range_overrides(method)
            if param_name is None:
                continue

            for override in overrides:
                label = override.get("label", "")
                run_label = f"{method} ({param_name}={label})"
                row_defs.append({
                    "row_key": run_label,
                    "method": method,
                    "param": param_name,
                    "value": label,
                    "is_base": False
                })

        return row_defs

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
        self.paused_rows.clear()
        self._build_live_grid_rows(self._base_grid_defs())
        self.agent.reset_stats()
        self._update_plot()
        self._log_status("Training reset")

    def _start_training(self):
        if self.training:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        selected_methods = self._get_selected_methods()
        if not selected_methods:
            messagebox.showwarning("Warning", "No methods available to train")
            return

        if self.animation_method_var.get() in selected_methods and not self.show_animation.get():
            self.show_animation.set(True)
            self._toggle_animation()

        if self.show_animation.get() and self.animation_method_var.get() not in selected_methods:
            self.animation_method_var.set(selected_methods[0])
            self._log_status("Animation method updated to match selected training method.")

        row_defs = self._build_training_grid_defs(selected_methods)
        self._build_live_grid_rows(row_defs)

        self.training = True

        for method_name in selected_methods:
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

            range_overrides, range_param = self._get_range_overrides(method_name)

            for override in range_overrides:
                if not self.training:
                    break

                label = override.get("label", "default")
                label = label.replace("default,", "").replace("default", "").strip(",") or "default"

                if range_param:
                    run_label = f"{method_name} ({range_param}={label})"
                else:
                    run_label = method_name if label == "default" else f"{method_name} ({label})"

                row_key = run_label

                if method_name in self.grid_rows:
                    base_row = self.grid_rows[method_name]
                    base_row["param_var"].set(range_param or "-")
                    base_row["value_var"].set("-" if label == "default" else label)

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

                    while row_key in self.paused_rows and self.training:
                        time.sleep(0.1)

                    render = self.show_animation.get() and self.animation_method_var.get() == method_name
                    render_callback = self._update_animation_frame if render else None

                    reward, steps = agent.run_episode(render=render, train=True, render_callback=render_callback)

                    returns = self.method_stats[run_label]["returns"]
                    returns.append(reward)

                    avg_reward = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
                    duration = time.time() - start_time

                    self._update_grid(row_key, episode + 1, episodes, steps, avg_reward, reward, duration)
                    if row_key != method_name:
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

    def _update_grid(self, row_key, episode, episodes, steps, avg_reward, reward, duration):
        def update():
            if row_key not in self.grid_rows:
                return
            row = self.grid_rows[row_key]
            row["episode_var"].set(f"{episode}/{episodes}")
            row["step_var"].set(str(steps))
            row["avg_var"].set(f"{avg_reward:.2f}")
            row["reward_var"].set(f"{reward:.2f}")
            row["duration_var"].set(self._format_duration(duration))

        if threading.current_thread() == threading.main_thread():
            update()
        else:
            self.root.after(0, update)

    def _run_episode(self):
        if self.training:
            messagebox.showwarning("Warning", "Stop training before running a single episode.")
            return

        selected_methods = self._get_selected_methods()
        if not selected_methods:
            messagebox.showwarning("Warning", "No methods available to run")
            return
        method_name = selected_methods[0]
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
        selected_methods = self._get_selected_methods()
        if not selected_methods:
            messagebox.showwarning("Warning", "No methods available to save")
            return
        method_name = selected_methods[0]
        value_label = ""
        base_row = self.grid_rows.get(method_name)
        if base_row:
            param = base_row["param_var"].get()
            value = base_row["value_var"].get()
            if param not in ("-", "") and value not in ("-", "default", ""):
                value_label = f"{param}={value}"
        value_part = "" if not value_label else f"_{value_label.replace(' ', '_')}"
        filename = f"pusher_plot_{method_name}{value_part}_{timestamp}.png"
        self.fig.savefig(filename, facecolor="#2b2b2b", edgecolor="white")
        self._log_status(f"Plot saved to {filename}")
        messagebox.showinfo("Success", f"Plot saved to {filename}")

    def _log_status(self, message):
        print(message)

    def run(self):
        self.root.mainloop()
