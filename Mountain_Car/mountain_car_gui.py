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


@dataclass
class MethodConfig:
    name: str
    policy_cls: object
    color_light: str
    color_bold: str


class MountainCarGUI:
    def __init__(self, master: tk.Tk, env: MountainCarEnv, agent: Agent, policies: List):
        self.master = master
        self.env = env
        self.agent = agent
        self.policies = policies
        self.master.title("Mountain Car RL Demo")

        self.methods = [
            MethodConfig("DQN", DQNPolicy, "#ffb3b3", "#ff4d4d"),
            MethodConfig("DDQN", DDQNPolicy, "#b3e6ff", "#3399ff"),
            MethodConfig("Prioritized DDQN", PrioritizedDDQNPolicy, "#d1b3ff", "#8a4dff"),
            MethodConfig("Dueling DDQN", DuelingDDQNPolicy, "#b3d1ff", "#4d79ff"),
            MethodConfig("Distributional DQN", DistributionalDQNPolicy, "#ffd9b3", "#ff944d"),
            MethodConfig("Noisy DQN", NoisyDQNPolicy, "#b3ffb3", "#4dff4d"),
            MethodConfig("Rainbow", RainbowPolicy, "#ffb3e6", "#ff4db8"),
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
        top_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

        control_frame = ttk.LabelFrame(self.master, text="Training Configuration")
        control_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

        method_frame = ttk.LabelFrame(self.master, text="Method Parameters")
        method_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)

        action_frame = ttk.LabelFrame(self.master, text="Action")
        action_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=6)

        plot_frame = ttk.Frame(self.master)
        plot_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=6)

        self.render_canvas = tk.Canvas(top_frame, width=600, height=200, bg="black")
        self.render_canvas.pack(fill="both", expand=True)

        ttk.Label(control_frame, text="Episodes").grid(row=0, column=0, sticky="w")
        self.episodes_var = tk.IntVar(value=500)
        ttk.Entry(control_frame, textvariable=self.episodes_var, width=8).grid(row=0, column=1)

        ttk.Label(control_frame, text="Alpha").grid(row=1, column=0, sticky="w")
        self.alpha_var = tk.DoubleVar(value=0.0005)
        ttk.Entry(control_frame, textvariable=self.alpha_var, width=8).grid(row=1, column=1)

        ttk.Label(control_frame, text="Gamma").grid(row=2, column=0, sticky="w")
        self.gamma_var = tk.DoubleVar(value=0.99)
        ttk.Entry(control_frame, textvariable=self.gamma_var, width=8).grid(row=2, column=1)

        ttk.Label(control_frame, text="Epsilon start").grid(row=3, column=0, sticky="w")
        self.eps_start_var = tk.DoubleVar(value=1.0)
        ttk.Entry(control_frame, textvariable=self.eps_start_var, width=8).grid(row=3, column=1)

        ttk.Label(control_frame, text="Epsilon end").grid(row=4, column=0, sticky="w")
        self.eps_end_var = tk.DoubleVar(value=0.02)
        ttk.Entry(control_frame, textvariable=self.eps_end_var, width=8).grid(row=4, column=1)

        ttk.Label(control_frame, text="Epsilon decay").grid(row=5, column=0, sticky="w")
        self.eps_decay_var = tk.DoubleVar(value=0.0005)
        ttk.Entry(control_frame, textvariable=self.eps_decay_var, width=8).grid(row=5, column=1)

        ttk.Label(control_frame, text="Step delay (s)").grid(row=6, column=0, sticky="w")
        self.step_delay_var = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.step_delay_var, width=8).grid(row=6, column=1)

        ttk.Label(control_frame, text="Max steps").grid(row=7, column=0, sticky="w")
        self.max_steps_var = tk.IntVar(value=200)
        ttk.Entry(control_frame, textvariable=self.max_steps_var, width=8).grid(row=7, column=1)

        ttk.Label(control_frame, text="Seed").grid(row=8, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(control_frame, textvariable=self.seed_var, width=8).grid(row=8, column=1)

        ttk.Label(control_frame, text="Render interval (s)").grid(row=9, column=0, sticky="w")
        self.render_interval_var = tk.DoubleVar(value=0.05)
        ttk.Entry(control_frame, textvariable=self.render_interval_var, width=8).grid(row=9, column=1)

        ttk.Label(control_frame, text="Goal velocity").grid(row=10, column=0, sticky="w")
        self.goal_velocity_var = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.goal_velocity_var, width=8).grid(row=10, column=1)

        ttk.Label(control_frame, text="x_init (optional)").grid(row=11, column=0, sticky="w")
        self.x_init_var = tk.StringVar(value=str(np.pi))
        ttk.Entry(control_frame, textvariable=self.x_init_var, width=8).grid(row=11, column=1)

        ttk.Label(control_frame, text="y_init (optional)").grid(row=12, column=0, sticky="w")
        self.y_init_var = tk.StringVar(value="1.0")
        ttk.Entry(control_frame, textvariable=self.y_init_var, width=8).grid(row=12, column=1)

        ttk.Label(control_frame, text="Environment").grid(row=13, column=0, sticky="w")
        self.env_var = tk.StringVar(value="Continuous")
        ttk.Combobox(
            control_frame,
            values=["Continuous", "Discrete"],
            textvariable=self.env_var,
            state="readonly",
            width=10,
        ).grid(row=13, column=1, sticky="w")

        ttk.Label(control_frame, text="Action bins").grid(row=14, column=0, sticky="w")
        self.action_bins_var = tk.IntVar(value=11)
        ttk.Entry(control_frame, textvariable=self.action_bins_var, width=8).grid(row=14, column=1)

        ttk.Label(control_frame, text="Reward scale").grid(row=15, column=0, sticky="w")
        self.reward_scale_var = tk.DoubleVar(value=1.0)
        ttk.Entry(control_frame, textvariable=self.reward_scale_var, width=8).grid(row=15, column=1)

        ttk.Label(control_frame, text="Grad clip").grid(row=16, column=0, sticky="w")
        self.grad_clip_var = tk.DoubleVar(value=10.0)
        ttk.Entry(control_frame, textvariable=self.grad_clip_var, width=8).grid(row=16, column=1)

        ttk.Label(control_frame, text="Hidden layers").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.hidden_layers_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.hidden_layers_var, width=8).grid(row=0, column=3)

        ttk.Label(control_frame, text="Hidden units").grid(row=1, column=2, sticky="w", padx=(12, 0))
        self.hidden_units_var = tk.IntVar(value=128)
        ttk.Entry(control_frame, textvariable=self.hidden_units_var, width=8).grid(row=1, column=3)

        ttk.Label(control_frame, text="Batch size").grid(row=2, column=2, sticky="w", padx=(12, 0))
        self.batch_size_var = tk.IntVar(value=128)
        ttk.Entry(control_frame, textvariable=self.batch_size_var, width=8).grid(row=2, column=3)

        ttk.Label(control_frame, text="Replay size").grid(row=3, column=2, sticky="w", padx=(12, 0))
        self.replay_size_var = tk.IntVar(value=50000)
        ttk.Entry(control_frame, textvariable=self.replay_size_var, width=8).grid(row=3, column=3)

        ttk.Label(control_frame, text="Warm-up steps").grid(row=4, column=2, sticky="w", padx=(12, 0))
        self.warmup_steps_var = tk.IntVar(value=2000)
        ttk.Entry(control_frame, textvariable=self.warmup_steps_var, width=8).grid(row=4, column=3)

        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Normalize states", variable=self.normalize_var).grid(row=5, column=2, columnspan=2, sticky="w", padx=(12, 0))

        ttk.Label(control_frame, text="Activation").grid(row=6, column=2, sticky="w", padx=(12, 0))
        self.activation_var = tk.StringVar(value="ReLU")
        ttk.Combobox(
            control_frame,
            values=["ReLU", "LeakyReLU", "Tanh", "ELU", "GELU"],
            textvariable=self.activation_var,
            state="readonly",
            width=8,
        ).grid(row=6, column=3, sticky="w")

        ttk.Label(control_frame, text="Method").grid(row=7, column=2, sticky="w", padx=(12, 0))
        ttk.Combobox(control_frame, values=[m.name for m in self.methods], textvariable=self.method_var, state="readonly").grid(row=7, column=3, sticky="w")

        ttk.Checkbutton(control_frame, text="Compare methods", variable=self.compare_methods).grid(row=8, column=2, columnspan=2, sticky="w", padx=(12, 0))

        dqn_frame = ttk.LabelFrame(method_frame, text="DQN")
        dqn_frame.grid(row=0, column=0, padx=6, pady=4, sticky="nsew")
        ttk.Label(dqn_frame, text="(no special params)").grid(row=0, column=0, sticky="w")

        ddqn_frame = ttk.LabelFrame(method_frame, text="DDQN")
        ddqn_frame.grid(row=0, column=1, padx=6, pady=4, sticky="nsew")
        ttk.Label(ddqn_frame, text="Target update").grid(row=0, column=0, sticky="w")
        self.ddqn_target_update_var = tk.IntVar(value=20)
        ttk.Entry(ddqn_frame, textvariable=self.ddqn_target_update_var, width=8).grid(row=0, column=1)

        prio_frame = ttk.LabelFrame(method_frame, text="Prioritized DDQN")
        prio_frame.grid(row=0, column=2, padx=6, pady=4, sticky="nsew")
        ttk.Label(prio_frame, text="Target update").grid(row=0, column=0, sticky="w")
        self.prio_target_update_var = tk.IntVar(value=20)
        ttk.Entry(prio_frame, textvariable=self.prio_target_update_var, width=8).grid(row=0, column=1)
        ttk.Label(prio_frame, text="Priority alpha").grid(row=1, column=0, sticky="w")
        self.prio_alpha_var = tk.DoubleVar(value=0.6)
        ttk.Entry(prio_frame, textvariable=self.prio_alpha_var, width=8).grid(row=1, column=1)
        ttk.Label(prio_frame, text="Beta start").grid(row=2, column=0, sticky="w")
        self.prio_beta_start_var = tk.DoubleVar(value=0.4)
        ttk.Entry(prio_frame, textvariable=self.prio_beta_start_var, width=8).grid(row=2, column=1)
        ttk.Label(prio_frame, text="Beta end").grid(row=3, column=0, sticky="w")
        self.prio_beta_end_var = tk.DoubleVar(value=1.0)
        ttk.Entry(prio_frame, textvariable=self.prio_beta_end_var, width=8).grid(row=3, column=1)
        ttk.Label(prio_frame, text="Beta steps").grid(row=4, column=0, sticky="w")
        self.prio_beta_steps_var = tk.IntVar(value=50000)
        ttk.Entry(prio_frame, textvariable=self.prio_beta_steps_var, width=8).grid(row=4, column=1)

        dueling_frame = ttk.LabelFrame(method_frame, text="Dueling DDQN")
        dueling_frame.grid(row=1, column=0, padx=6, pady=4, sticky="nsew")
        ttk.Label(dueling_frame, text="Target update").grid(row=0, column=0, sticky="w")
        self.target_update_var = tk.IntVar(value=20)
        ttk.Entry(dueling_frame, textvariable=self.target_update_var, width=8).grid(row=0, column=1)

        dist_frame = ttk.LabelFrame(method_frame, text="Distributional DQN")
        dist_frame.grid(row=1, column=1, padx=6, pady=4, sticky="nsew")
        ttk.Label(dist_frame, text="Atoms").grid(row=0, column=0, sticky="w")
        self.atoms_var = tk.IntVar(value=51)
        ttk.Entry(dist_frame, textvariable=self.atoms_var, width=8).grid(row=0, column=1)
        ttk.Label(dist_frame, text="Vmin").grid(row=1, column=0, sticky="w")
        self.vmin_var = tk.DoubleVar(value=-200.0)
        ttk.Entry(dist_frame, textvariable=self.vmin_var, width=8).grid(row=1, column=1)
        ttk.Label(dist_frame, text="Vmax").grid(row=2, column=0, sticky="w")
        self.vmax_var = tk.DoubleVar(value=0.0)
        ttk.Entry(dist_frame, textvariable=self.vmax_var, width=8).grid(row=2, column=1)
        ttk.Label(dist_frame, text="Target update").grid(row=3, column=0, sticky="w")
        self.dist_target_update_var = tk.IntVar(value=20)
        ttk.Entry(dist_frame, textvariable=self.dist_target_update_var, width=8).grid(row=3, column=1)

        noisy_frame = ttk.LabelFrame(method_frame, text="Noisy DQN")
        noisy_frame.grid(row=1, column=2, padx=6, pady=4, sticky="nsew")
        ttk.Label(noisy_frame, text="Noisy sigma").grid(row=0, column=0, sticky="w")
        self.noisy_sigma_var = tk.DoubleVar(value=0.5)
        ttk.Entry(noisy_frame, textvariable=self.noisy_sigma_var, width=8).grid(row=0, column=1)

        rainbow_frame = ttk.LabelFrame(method_frame, text="Rainbow")
        rainbow_frame.grid(row=2, column=0, columnspan=3, padx=6, pady=4, sticky="nsew")
        ttk.Label(rainbow_frame, text="Atoms").grid(row=0, column=0, sticky="w")
        self.rainbow_atoms_var = tk.IntVar(value=51)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_atoms_var, width=8).grid(row=0, column=1)
        ttk.Label(rainbow_frame, text="Vmin").grid(row=0, column=2, sticky="w")
        self.rainbow_vmin_var = tk.DoubleVar(value=-200.0)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_vmin_var, width=8).grid(row=0, column=3)
        ttk.Label(rainbow_frame, text="Vmax").grid(row=0, column=4, sticky="w")
        self.rainbow_vmax_var = tk.DoubleVar(value=0.0)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_vmax_var, width=8).grid(row=0, column=5)
        ttk.Label(rainbow_frame, text="Target update").grid(row=1, column=0, sticky="w")
        self.rainbow_target_update_var = tk.IntVar(value=20)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_target_update_var, width=8).grid(row=1, column=1)
        ttk.Label(rainbow_frame, text="Noisy sigma").grid(row=1, column=2, sticky="w")
        self.rainbow_noisy_sigma_var = tk.DoubleVar(value=0.5)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_noisy_sigma_var, width=8).grid(row=1, column=3)
        ttk.Label(rainbow_frame, text="Priority alpha").grid(row=1, column=4, sticky="w")
        self.rainbow_prio_alpha_var = tk.DoubleVar(value=0.6)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_prio_alpha_var, width=8).grid(row=1, column=5)
        ttk.Label(rainbow_frame, text="Beta start").grid(row=2, column=0, sticky="w")
        self.rainbow_beta_start_var = tk.DoubleVar(value=0.4)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_beta_start_var, width=8).grid(row=2, column=1)
        ttk.Label(rainbow_frame, text="Beta end").grid(row=2, column=2, sticky="w")
        self.rainbow_beta_end_var = tk.DoubleVar(value=1.0)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_beta_end_var, width=8).grid(row=2, column=3)
        ttk.Label(rainbow_frame, text="Beta steps").grid(row=2, column=4, sticky="w")
        self.rainbow_beta_steps_var = tk.IntVar(value=50000)
        ttk.Entry(rainbow_frame, textvariable=self.rainbow_beta_steps_var, width=8).grid(row=2, column=5)

        ttk.Button(action_frame, text="Reset", command=self.reset_ui).grid(row=0, column=0, padx=4)
        ttk.Button(action_frame, text="Train and run", command=self.train_and_run).grid(row=0, column=1, padx=4)
        ttk.Button(action_frame, text="Cancel learning", command=self.cancel_learning).grid(row=0, column=2, padx=4)
        ttk.Button(action_frame, text="Save plot", command=self.save_plot).grid(row=0, column=3, padx=4)

        self.anim_btn = ttk.Button(action_frame, text="Disable animation", command=self.toggle_animation)
        self.anim_btn.grid(row=1, column=0, padx=4, pady=(4, 0))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(action_frame, textvariable=self.status_var).grid(row=1, column=1, columnspan=3, sticky="w", pady=(4, 0))

        self.fig = Figure(figsize=(8, 4.5), dpi=100)
        self.fig.patch.set_facecolor("#111111")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#111111")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("#888888")
        self.ax.set_title("Episode Returns", color="white")
        self.ax.set_xlabel("Episode", color="white")
        self.ax.set_ylabel("Return", color="white")
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

    def toggle_animation(self):
        self._animation_enabled = not self._animation_enabled
        self.anim_btn.config(text="Disable animation" if self._animation_enabled else "Enable animation")

    def reset_ui(self):
        self.cancel_learning()
        self.returns_by_method = {}
        self._render_method = None
        self._update_plot()
        self.status_var.set("Idle")

    def _schedule_queue_poll(self):
        self.master.after(50, self._poll_queue)

    def _poll_queue(self):
        updated_plot = False
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                break

            msg_type = msg.get("type")
            if msg_type == "reward":
                name = msg["method"]
                value = msg["value"]
                if name not in self.returns_by_method:
                    self.returns_by_method[name] = []
                self.returns_by_method[name].append(value)
                updated_plot = True
            elif msg_type == "frame" and msg.get("frame") is not None:
                self._update_render(msg["frame"])
            elif msg_type == "done":
                method = msg.get("method")
                if method in self._workers:
                    self._workers.pop(method, None)
                if not self._workers:
                    self.status_var.set("Idle")
            elif msg_type == "error":
                method = msg.get("method", "Unknown")
                err = msg.get("error", "Unknown error")
                self.status_var.set(f"Error in {method}: {err}")
                if method in self._workers:
                    self._workers.pop(method, None)

        if updated_plot:
            self._update_plot()

        self._schedule_queue_poll()

    def _update_render(self, frame):
        if frame is None:
            return
        img = Image.fromarray(frame)
        canvas_w = self.render_canvas.winfo_width() or 600
        canvas_h = self.render_canvas.winfo_height() or 200
        img = img.resize((canvas_w, canvas_h))
        self._render_image = ImageTk.PhotoImage(img)
        self.render_canvas.delete("all")
        self.render_canvas.create_image(0, 0, anchor="nw", image=self._render_image)

    def _moving_average(self, values: List[float], window: int = 20) -> List[float]:
        if len(values) < window:
            return []
        arr = np.array(values, dtype=np.float32)
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(arr, kernel, mode="valid").tolist()

    def _update_plot(self):
        self.ax.cla()
        self.ax.set_facecolor("#111111")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("#888888")
        for m in self.methods:
            name = m.name
            vals = self.returns_by_method.get(name, [])
            if not vals:
                continue
            self.ax.plot(vals, color=m.color_light, lw=1, label=f"{name} returns")
            ma = self._moving_average(vals)
            if ma:
                x = list(range(len(vals) - len(ma), len(vals)))
                self.ax.plot(x, ma, color=m.color_bold, lw=2, label=f"{name} avg")
        self.ax.set_title("Episode Returns", color="white")
        self.ax.set_xlabel("Episode", color="white")
        self.ax.set_ylabel("Return", color="white")
        self.ax.grid(color="#333333", linestyle="--", linewidth=0.5, alpha=0.7)
        legend = self.ax.legend(facecolor="#111111", edgecolor="#444444")
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("white")
        self.canvas_plot.draw_idle()

    def train_and_run(self):
        selected_method = self.method_var.get()
        if not selected_method:
            self.status_var.set("Select a method")
            return

        if self.compare_methods.get():
            selected = [m.name for m in self.methods]
        else:
            selected = [selected_method]

        self.cancel_learning()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.returns_by_method = {name: [] for name in selected}
        self._render_method = selected_method
        self.status_var.set("Training...")

        for name in selected:
            enable_render = self._animation_enabled and name == self._render_method
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._train_worker,
                args=(name, stop_event, enable_render),
                daemon=True,
            )
            self._workers[name] = {"thread": thread, "stop": stop_event}
            thread.start()

    def cancel_learning(self):
        if not self._workers:
            return
        for info in self._workers.values():
            info["stop"].set()
        self._workers = {}
        self.status_var.set("Stopping...")

    def _parse_optional_float(self, text: str) -> Optional[float]:
        stripped = text.strip()
        if stripped == "":
            return None
        return float(stripped)

    def _train_worker(self, method_name: str, stop_event: threading.Event, enable_render: bool):
        try:
            episodes = int(self.episodes_var.get())
            alpha = float(self.alpha_var.get())
            gamma = float(self.gamma_var.get())
            eps_start = float(self.eps_start_var.get())
            eps_end = float(self.eps_end_var.get())
            eps_decay = float(self.eps_decay_var.get())
            step_delay = float(self.step_delay_var.get())
            max_steps = int(self.max_steps_var.get())
            hidden_layers = int(self.hidden_layers_var.get())
            hidden_units = int(self.hidden_units_var.get())
            batch_size = int(self.batch_size_var.get())
            replay_size = int(self.replay_size_var.get())
            warmup_steps = int(self.warmup_steps_var.get())
            normalize_states = bool(self.normalize_var.get())
            reward_scale = float(self.reward_scale_var.get())
            grad_clip = float(self.grad_clip_var.get())
            activation = self.activation_var.get()
            render_interval = float(self.render_interval_var.get())
            goal_velocity = float(self.goal_velocity_var.get())
            action_bins = max(2, int(self.action_bins_var.get()))
            env_choice = self.env_var.get()
            env_name = "MountainCarContinuous-v0" if env_choice == "Continuous" else "MountainCar-v0"

            seed_text = self.seed_var.get().strip()
            seed = int(seed_text) if seed_text else None
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                import torch
                torch.manual_seed(seed)

            x_init = self._parse_optional_float(self.x_init_var.get())
            y_init = self._parse_optional_float(self.y_init_var.get())

            render_mode = "rgb_array" if enable_render else None
            env = MountainCarEnv(
                seed=seed,
                render_mode=render_mode,
                env_name=env_name,
                goal_velocity=goal_velocity,
                x_init=x_init,
                y_init=y_init,
                action_bins=action_bins,
            )
            obs_shape = env.observation_space.shape
            if obs_shape is None:
                env.close()
                raise ValueError("MountainCar observation space must be a Box with shape.")
            state_dim = int(obs_shape[0])
            action_dim = env.action_bins

            policy_map = {
                "DQN": DQNPolicy,
                "DDQN": DDQNPolicy,
                "Prioritized DDQN": PrioritizedDDQNPolicy,
                "Dueling DDQN": DuelingDDQNPolicy,
                "Distributional DQN": DistributionalDQNPolicy,
                "Noisy DQN": NoisyDQNPolicy,
                "Rainbow": RainbowPolicy,
            }
            policy_cls = policy_map[method_name]

            if method_name == "DDQN":
                target_update = int(self.ddqn_target_update_var.get())
                policy = policy_cls(
                    state_dim,
                    action_dim,
                    alpha,
                    gamma,
                    eps_start,
                    eps_end,
                    eps_decay,
                    hidden_layers,
                    hidden_units,
                    batch_size,
                    replay_size,
                    activation,
                    warmup_steps=warmup_steps,
                    normalize_states=normalize_states,
                    reward_scale=reward_scale,
                    grad_clip=grad_clip,
                    target_update=target_update,
                )
            elif method_name == "Prioritized DDQN":
                target_update = int(self.prio_target_update_var.get())
                prior_alpha = float(self.prio_alpha_var.get())
                beta_start = float(self.prio_beta_start_var.get())
                beta_end = float(self.prio_beta_end_var.get())
                beta_steps = int(self.prio_beta_steps_var.get())
                policy = policy_cls(
                    state_dim,
                    action_dim,
                    alpha,
                    gamma,
                    eps_start,
                    eps_end,
                    eps_decay,
                    hidden_layers,
                    hidden_units,
                    batch_size,
                    replay_size,
                    activation,
                    warmup_steps=warmup_steps,
                    normalize_states=normalize_states,
                    reward_scale=reward_scale,
                    grad_clip=grad_clip,
                    target_update=target_update,
                    prior_alpha=prior_alpha,
                    beta_start=beta_start,
                    beta_end=beta_end,
                    beta_steps=beta_steps,
                )
            elif method_name == "Dueling DDQN":
                target_update = int(self.target_update_var.get())
                policy = policy_cls(
                    state_dim,
                    action_dim,
                    alpha,
                    gamma,
                    eps_start,
                    eps_end,
                    eps_decay,
                    hidden_layers,
                    hidden_units,
                    batch_size,
                    replay_size,
                    activation,
                    warmup_steps=warmup_steps,
                    normalize_states=normalize_states,
                    reward_scale=reward_scale,
                    grad_clip=grad_clip,
                    target_update=target_update,
                )
            elif method_name == "Noisy DQN":
                noisy_sigma = float(self.noisy_sigma_var.get())
                policy = policy_cls(
                    state_dim,
                    action_dim,
                    alpha,
                    gamma,
                    eps_start,
                    eps_end,
                    eps_decay,
                    hidden_layers,
                    hidden_units,
                    batch_size,
                    replay_size,
                    activation,
                    warmup_steps=warmup_steps,
                    normalize_states=normalize_states,
                    reward_scale=reward_scale,
                    grad_clip=grad_clip,
                    noisy_sigma=noisy_sigma,
                )
            elif method_name == "Distributional DQN":
                atoms = int(self.atoms_var.get())
                vmin = float(self.vmin_var.get())
                vmax = float(self.vmax_var.get())
                target_update = int(self.dist_target_update_var.get())
                policy = policy_cls(
                    state_dim,
                    action_dim,
                    alpha,
                    gamma,
                    eps_start,
                    eps_end,
                    eps_decay,
                    hidden_layers,
                    hidden_units,
                    batch_size,
                    replay_size,
                    activation,
                    warmup_steps=warmup_steps,
                    normalize_states=normalize_states,
                    reward_scale=reward_scale,
                    grad_clip=grad_clip,
                    atoms=atoms,
                    v_min=vmin,
                    v_max=vmax,
                    target_update=target_update,
                )
            elif method_name == "Rainbow":
                atoms = int(self.rainbow_atoms_var.get())
                vmin = float(self.rainbow_vmin_var.get())
                vmax = float(self.rainbow_vmax_var.get())
                target_update = int(self.rainbow_target_update_var.get())
                noisy_sigma = float(self.rainbow_noisy_sigma_var.get())
                prior_alpha = float(self.rainbow_prio_alpha_var.get())
                beta_start = float(self.rainbow_beta_start_var.get())
                beta_end = float(self.rainbow_beta_end_var.get())
                beta_steps = int(self.rainbow_beta_steps_var.get())
                policy = policy_cls(
                    state_dim,
                    action_dim,
                    alpha,
                    gamma,
                    eps_start,
                    eps_end,
                    eps_decay,
                    hidden_layers,
                    hidden_units,
                    batch_size,
                    replay_size,
                    activation,
                    warmup_steps=warmup_steps,
                    normalize_states=normalize_states,
                    reward_scale=reward_scale,
                    grad_clip=grad_clip,
                    atoms=atoms,
                    v_min=vmin,
                    v_max=vmax,
                    target_update=target_update,
                    noisy_sigma=noisy_sigma,
                    prior_alpha=prior_alpha,
                    beta_start=beta_start,
                    beta_end=beta_end,
                    beta_steps=beta_steps,
                )
            else:
                policy = policy_cls(
                    state_dim,
                    action_dim,
                    alpha,
                    gamma,
                    eps_start,
                    eps_end,
                    eps_decay,
                    hidden_layers,
                    hidden_units,
                    batch_size,
                    replay_size,
                    activation,
                    warmup_steps=warmup_steps,
                    normalize_states=normalize_states,
                    reward_scale=reward_scale,
                    grad_clip=grad_clip,
                )

            agent = Agent(env, policy)

            last_frame_time = 0.0

            def render_cb(frame):
                nonlocal last_frame_time
                now = time.monotonic()
                if now - last_frame_time >= render_interval:
                    self._queue.put({"type": "frame", "frame": frame})
                    last_frame_time = now

            for ep in range(episodes):
                if stop_event.is_set():
                    break
                total = agent.run_episode(ep, max_steps, step_delay, render_cb if enable_render else None, stop_event)
                self._queue.put({"type": "reward", "method": method_name, "value": total})

            env.close()
            self._queue.put({"type": "done", "method": method_name})
        except Exception as exc:
            self._queue.put({"type": "error", "method": method_name, "error": str(exc)})

    def save_plot(self):
        name = self._render_method or "compare"
        default_name = f"plot_{name.replace(' ', '_')}.png"
        file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_name)
        if not file_path:
            return
        self.fig.savefig(file_path)
