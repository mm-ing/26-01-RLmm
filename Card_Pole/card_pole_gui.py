import queue
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

from card_pole_logic import (
    Agent,
    CartPoleEnv,
    DQNPolicy,
    DDQNPolicy,
)


@dataclass
class MethodConfig:
    name: str
    policy_cls: object
    color_light: str
    color_bold: str


class CardPoleGUI:
    def __init__(self, master: tk.Tk, env: CartPoleEnv, agent: Agent, policies: List):
        self.master = master
        self.env = env
        self.agent = agent
        self.policies = policies
        self.master.title("CartPole RL Demo")

        self.methods = [
            MethodConfig("DQN", DQNPolicy, "#ffb3b3", "#ff4d4d"),
            MethodConfig("DDQN", DDQNPolicy, "#b3d1ff", "#4d79ff"),
        ]

        self.returns_by_method: Dict[str, List[float]] = {}
        self.compare_methods = tk.BooleanVar(value=True)
        self.method_var = tk.StringVar(value=self.methods[0].name)
        self._render_method: Optional[str] = None

        self._workers: Dict[str, Dict] = {}
        self._queue: queue.Queue = queue.Queue()
        self._render_image: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._schedule_queue_poll()

    def _build_ui(self):
        top_frame = ttk.Frame(self.master)
        top_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

        control_frame = ttk.LabelFrame(self.master, text="Training Configuration")
        control_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

        action_frame = ttk.LabelFrame(self.master, text="Action")
        action_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)

        plot_frame = ttk.Frame(self.master)
        plot_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=6)

        self.render_canvas = tk.Canvas(top_frame, width=600, height=200, bg="black")
        self.render_canvas.pack(fill="both", expand=True)

        ttk.Label(control_frame, text="Episodes").grid(row=0, column=0, sticky="w")
        self.episodes_var = tk.IntVar(value=1000)
        ttk.Entry(control_frame, textvariable=self.episodes_var, width=8).grid(row=0, column=1)

        ttk.Label(control_frame, text="Alpha").grid(row=1, column=0, sticky="w")
        self.alpha_var = tk.DoubleVar(value=0.2)
        ttk.Entry(control_frame, textvariable=self.alpha_var, width=8).grid(row=1, column=1)

        ttk.Label(control_frame, text="Gamma").grid(row=2, column=0, sticky="w")
        self.gamma_var = tk.DoubleVar(value=0.8)
        ttk.Entry(control_frame, textvariable=self.gamma_var, width=8).grid(row=2, column=1)

        ttk.Label(control_frame, text="Epsilon start").grid(row=3, column=0, sticky="w")
        self.eps_start_var = tk.DoubleVar(value=1.0)
        ttk.Entry(control_frame, textvariable=self.eps_start_var, width=8).grid(row=3, column=1)

        ttk.Label(control_frame, text="Epsilon end").grid(row=4, column=0, sticky="w")
        self.eps_end_var = tk.DoubleVar(value=0.05)
        ttk.Entry(control_frame, textvariable=self.eps_end_var, width=8).grid(row=4, column=1)

        ttk.Label(control_frame, text="Epsilon decay").grid(row=5, column=0, sticky="w")
        self.eps_decay_var = tk.DoubleVar(value=0.05)
        ttk.Entry(control_frame, textvariable=self.eps_decay_var, width=8).grid(row=5, column=1)

        ttk.Label(control_frame, text="Step delay (s)").grid(row=6, column=0, sticky="w")
        self.step_delay_var = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.step_delay_var, width=8).grid(row=6, column=1)

        ttk.Label(control_frame, text="Max steps").grid(row=7, column=0, sticky="w")
        self.max_steps_var = tk.IntVar(value=500)
        ttk.Entry(control_frame, textvariable=self.max_steps_var, width=8).grid(row=7, column=1)

        ttk.Label(control_frame, text="Hidden layers").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.hidden_layers_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.hidden_layers_var, width=8).grid(row=0, column=3)

        ttk.Label(control_frame, text="Hidden units").grid(row=1, column=2, sticky="w", padx=(12, 0))
        self.hidden_units_var = tk.IntVar(value=64)
        ttk.Entry(control_frame, textvariable=self.hidden_units_var, width=8).grid(row=1, column=3)

        ttk.Label(control_frame, text="Batch size").grid(row=2, column=2, sticky="w", padx=(12, 0))
        self.batch_size_var = tk.IntVar(value=64)
        ttk.Entry(control_frame, textvariable=self.batch_size_var, width=8).grid(row=2, column=3)

        ttk.Label(control_frame, text="Replay size").grid(row=3, column=2, sticky="w", padx=(12, 0))
        self.replay_size_var = tk.IntVar(value=2000)
        ttk.Entry(control_frame, textvariable=self.replay_size_var, width=8).grid(row=3, column=3)

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

        ttk.Button(action_frame, text="Train and run", command=self.train_and_run).grid(row=0, column=0, padx=4)
        ttk.Button(action_frame, text="Cancel learning", command=self.cancel_learning).grid(row=0, column=1, padx=4)
        ttk.Button(action_frame, text="Save plot", command=self.save_plot).grid(row=0, column=2, padx=4)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(action_frame, textvariable=self.status_var).grid(row=1, column=0, columnspan=3, sticky="w")

        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Episode Returns")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Return")
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

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
                self.status_var.set("Idle")

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
        self.ax.set_title("Episode Returns")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Return")
        self.ax.legend()
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
        self.returns_by_method = {name: [] for name in selected}
        self._render_method = selected_method
        self.status_var.set("Training...")

        for name in selected:
            enable_render = name == self._render_method
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._train_worker,
                args=(name, stop_event, enable_render),
                daemon=True,
            )
            thread.start()
            self._workers[name] = {"thread": thread, "stop": stop_event}

    def cancel_learning(self):
        for data in self._workers.values():
            data["stop"].set()
        self._workers = {}
        self.status_var.set("Cancelled")

    def _train_worker(self, method_name: str, stop_event: threading.Event, enable_render: bool):
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
        activation = self.activation_var.get()

        render_mode = "rgb_array" if enable_render else None
        env = CartPoleEnv(render_mode=render_mode)
        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)

        policy_map = {
            "DQN": DQNPolicy,
            "DDQN": DDQNPolicy,
        }
        policy_cls = policy_map[method_name]
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
        )
        agent = Agent(env, policy)

        last_frame_time = 0.0

        def render_cb(frame):
            nonlocal last_frame_time
            now = time.monotonic()
            if now - last_frame_time >= 0.05:
                self._queue.put({"type": "frame", "frame": frame})
                last_frame_time = now

        for ep in range(episodes):
            if stop_event.is_set():
                break
            total = agent.run_episode(ep, max_steps, step_delay, render_cb if enable_render else None, stop_event)
            self._queue.put({"type": "reward", "method": method_name, "value": total})

        env.close()
        self._queue.put({"type": "done", "method": method_name})

    def save_plot(self):
        name = self._render_method or "compare"
        default_name = f"plot_{name.replace(' ', '_')}.png"
        file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_name)
        if not file_path:
            return
        self.fig.savefig(file_path)
