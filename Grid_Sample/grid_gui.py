import csv
import datetime as dt
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Dict, List, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from grid_logic import (
    Agent,
    GridWorld,
    MonteCarloPolicies,
    SarsaPolicies,
    ExpSarsaPolicies,
    QlearningPolicies,
    a_star_path,
    bfs_path,
    path_to_actions,
    random_path,
)

State = Tuple[int, int]


class GridGUI:
    def __init__(self, master: tk.Tk, env: GridWorld, agent: Agent, policies: List):
        self.master = master
        self.env = env
        self.agent = agent
        self.policies = policies
        self.policy_map = {p.name: p for p in policies}
        self.current_policy_name = policies[0].name if policies else ""

        self.master.title("GridWorld RL Demo")

        # state
        self.path_history: List[State] = [self.env.start]
        self.returns_by_policy: Dict[str, List[float]] = {}
        self.compare_methods = tk.BooleanVar(value=True)

        # colors
        self.method_colors_light = {
            "Monte Carlo": "#ffb3b3",
            "SARSA": "#b3ffb3",
            "Expected SARSA": "#b3d1ff",
            "Q-learning": "#ffd9b3",
        }
        self.method_colors_bold = {
            "Monte Carlo": "#ff4d4d",
            "SARSA": "#4dff4d",
            "Expected SARSA": "#4d79ff",
            "Q-learning": "#ff944d",
        }

        self._build_ui()
        self._draw_grid()

    def _build_ui(self):
        # frames
        grid_cfg = ttk.LabelFrame(self.master, text="Grid Configuration")
        grid_cfg.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

        epi_cfg = ttk.LabelFrame(self.master, text="Episode Configuration")
        epi_cfg.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

        action_cfg = ttk.LabelFrame(self.master, text="Action")
        action_cfg.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)

        self.grid_canvas = tk.Canvas(self.master, width=600, height=400, bg="white")
        self.grid_canvas.grid(row=0, column=1, rowspan=3, padx=8, pady=8)

        # grid config
        ttk.Label(grid_cfg, text="Grid Size (W x H)").grid(row=0, column=0, sticky="w")
        self.grid_w_var = tk.IntVar(value=self.env.width)
        self.grid_h_var = tk.IntVar(value=self.env.height)
        ttk.Entry(grid_cfg, textvariable=self.grid_w_var, width=5).grid(row=0, column=1)
        ttk.Entry(grid_cfg, textvariable=self.grid_h_var, width=5).grid(row=0, column=2)

        ttk.Label(grid_cfg, text="Start (X,Y)").grid(row=1, column=0, sticky="w")
        self.start_var = tk.StringVar(value=f"{self.env.start[0]},{self.env.start[1]}")
        ttk.Entry(grid_cfg, textvariable=self.start_var, width=10).grid(row=1, column=1, columnspan=2)

        ttk.Label(grid_cfg, text="Goal (X,Y)").grid(row=2, column=0, sticky="w")
        self.goal_var = tk.StringVar(value=f"{self.env.goal[0]},{self.env.goal[1]}")
        ttk.Entry(grid_cfg, textvariable=self.goal_var, width=10).grid(row=2, column=1, columnspan=2)

        ttk.Label(grid_cfg, text="Blocked ((x,y);...)").grid(row=3, column=0, sticky="w")
        self.blocked_var = tk.StringVar(value="2,1;2,2")
        ttk.Entry(grid_cfg, textvariable=self.blocked_var, width=16).grid(row=3, column=1, columnspan=2)

        # episode config
        ttk.Label(epi_cfg, text="Episodes").grid(row=0, column=0, sticky="w")
        self.episodes_var = tk.IntVar(value=100)
        ttk.Entry(epi_cfg, textvariable=self.episodes_var, width=8).grid(row=0, column=1)

        ttk.Label(epi_cfg, text="Max steps").grid(row=1, column=0, sticky="w")
        self.max_steps_var = tk.IntVar(value=self.env.max_steps)
        ttk.Entry(epi_cfg, textvariable=self.max_steps_var, width=8).grid(row=1, column=1)

        ttk.Label(epi_cfg, text="Alpha").grid(row=2, column=0, sticky="w")
        self.alpha_var = tk.DoubleVar(value=0.1)
        ttk.Entry(epi_cfg, textvariable=self.alpha_var, width=8).grid(row=2, column=1)

        ttk.Label(epi_cfg, text="Gamma").grid(row=3, column=0, sticky="w")
        self.gamma_var = tk.DoubleVar(value=0.9)
        ttk.Entry(epi_cfg, textvariable=self.gamma_var, width=8).grid(row=3, column=1)

        ttk.Label(epi_cfg, text="Epsilon max").grid(row=4, column=0, sticky="w")
        self.eps_max_var = tk.DoubleVar(value=1.0)
        ttk.Entry(epi_cfg, textvariable=self.eps_max_var, width=8).grid(row=4, column=1)

        ttk.Label(epi_cfg, text="Epsilon min").grid(row=5, column=0, sticky="w")
        self.eps_min_var = tk.DoubleVar(value=0.05)
        ttk.Entry(epi_cfg, textvariable=self.eps_min_var, width=8).grid(row=5, column=1)

        ttk.Label(epi_cfg, text="Policy").grid(row=6, column=0, sticky="w")
        self.policy_var = tk.StringVar(value=self.current_policy_name)
        policy_names = [p.name for p in self.policies]
        ttk.Combobox(epi_cfg, values=policy_names, textvariable=self.policy_var, state="readonly").grid(row=6, column=1)

        ttk.Label(epi_cfg, text="Pathfinder").grid(row=7, column=0, sticky="w")
        self.path_var = tk.StringVar(value="RL policy")
        ttk.Combobox(epi_cfg, values=["RL policy", "A*", "BFS", "Random"], textvariable=self.path_var, state="readonly").grid(row=7, column=1)

        ttk.Checkbutton(epi_cfg, text="Compare methods", variable=self.compare_methods).grid(row=8, column=0, columnspan=2, sticky="w")

        apply_btn = tk.Button(epi_cfg, text="Apply and reset", bg="yellow", command=self.apply_and_reset)
        apply_btn.grid(row=9, column=0, columnspan=2, pady=4)

        # action panel
        ttk.Button(action_cfg, text="Run single step", command=self.run_step).grid(row=0, column=0, padx=4)
        ttk.Button(action_cfg, text="Run single episode", command=self.run_episode).grid(row=0, column=1, padx=4)
        ttk.Button(action_cfg, text="Train and run", command=self.train).grid(row=0, column=2, padx=4)

        self.progress = ttk.Progressbar(action_cfg, orient="horizontal", length=200, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=3, pady=4)

        ttk.Button(action_cfg, text="Value-Table", command=self.show_value_table).grid(row=2, column=0, padx=4)
        ttk.Button(action_cfg, text="Q-Table", command=self.show_q_table).grid(row=2, column=1, padx=4)
        ttk.Button(action_cfg, text="Save plot", command=self.save_plot).grid(row=2, column=2, padx=4)

        # plot
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Episode Returns")
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_plot.get_tk_widget().grid(row=3, column=0, columnspan=2, padx=8, pady=8, sticky="nsew")

    def apply_and_reset(self):
        self._apply_env_params()
        self._recreate_policies()
        self.returns_by_policy = {}
        self.path_history = [self.env.start]
        self._draw_grid()
        self._update_plot()

    def _apply_env_params(self):
        width = max(2, int(self.grid_w_var.get()))
        height = max(2, int(self.grid_h_var.get()))
        start = self._parse_xy(self.start_var.get(), default=(0, 0))
        goal = self._parse_xy(self.goal_var.get(), default=(width - 1, height - 1))
        blocked = self._parse_blocked(self.blocked_var.get())

        # auto-correct bounds
        start = (min(max(0, start[0]), width - 1), min(max(0, start[1]), height - 1))
        goal = (min(max(0, goal[0]), width - 1), min(max(0, goal[1]), height - 1))
        blocked = [(x, y) for (x, y) in blocked if 0 <= x < width and 0 <= y < height and (x, y) not in (start, goal)]

        self.env = GridWorld(width=width, height=height, start=start, goal=goal, blocked=blocked, max_steps=int(self.max_steps_var.get()))
        self.agent.env = self.env

    def _recreate_policies(self):
        alpha = float(self.alpha_var.get())
        gamma = float(self.gamma_var.get())
        eps_max = float(self.eps_max_var.get())
        eps_min = float(self.eps_min_var.get())
        self.policies = [
            MonteCarloPolicies(epsilon_min=eps_min),
            SarsaPolicies(alpha=alpha, gamma=gamma, epsilon_max=eps_max, epsilon_min=eps_min),
            ExpSarsaPolicies(alpha=alpha, gamma=gamma, epsilon_max=eps_max, epsilon_min=eps_min),
            QlearningPolicies(alpha=alpha, gamma=gamma, epsilon_max=eps_max, epsilon_min=eps_min),
        ]
        self.policy_map = {p.name: p for p in self.policies}
        self.current_policy_name = self.policy_var.get() if self.policy_var.get() in self.policy_map else self.policies[0].name
        self.agent.set_policy(self.policy_map[self.current_policy_name])

    def run_step(self):
        self.agent.policy.start_episode(self.agent.episode_idx)
        state = self.env.state
        action = self.agent.policy.select_action(state, self.env.available_actions())
        next_state, reward, done, _ = self.env.step(action)
        self.agent.policy.update(state, action, reward, next_state, done)
        self.agent.last_trajectory.append({
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        })
        self.path_history.append(next_state)
        if done:
            self.agent.policy.end_episode(self.agent.last_trajectory)
            self.agent.episode_idx += 1
        self._draw_grid()

    def run_episode(self):
        self._recreate_policies()
        if self.path_var.get() != "RL policy":
            self._run_pathfinder_episode()
        else:
            total, _ = self.agent.run_episode(max_steps=self.max_steps_var.get())
            self._record_return(self.current_policy_name, total)
        self._draw_grid()
        self._update_plot()

    def train(self):
        self._recreate_policies()
        episodes = int(self.episodes_var.get())
        if self.compare_methods.get():
            for p in self.policies:
                self.agent.set_policy(p)
                self.current_policy_name = p.name
                for ep in range(episodes):
                    total, _ = self.agent.run_episode(max_steps=self.max_steps_var.get())
                    self._record_return(p.name, total)
                    self._update_progress(ep + 1, episodes)
        else:
            p = self.policy_map[self.current_policy_name]
            self.agent.set_policy(p)
            for ep in range(episodes):
                total, _ = self.agent.run_episode(max_steps=self.max_steps_var.get())
                self._record_return(self.current_policy_name, total)
                self._update_progress(ep + 1, episodes)
        self._update_plot()
        self._draw_grid()

    def _run_pathfinder_episode(self):
        self.env.reset()
        start = self.env.start
        goal = self.env.goal
        max_steps = int(self.max_steps_var.get())
        if self.path_var.get() == "A*":
            path = a_star_path(self.env, start, goal)
        elif self.path_var.get() == "BFS":
            path = bfs_path(self.env, start, goal)
        else:
            path = random_path(self.env, start, goal, max_steps)
        actions = path_to_actions(path)
        self.agent.last_trajectory = []
        total = 0.0
        for a in actions[:max_steps]:
            s = self.env.state
            nxt, r, done, _ = self.env.step(a)
            self.agent.last_trajectory.append({
                "state": s,
                "action": a,
                "next_state": nxt,
                "reward": r,
                "done": done,
            })
            total += r
            if done:
                break
        self._record_return(self.current_policy_name, total)

    def _record_return(self, name: str, value: float):
        if name not in self.returns_by_policy:
            self.returns_by_policy[name] = []
        self.returns_by_policy[name].append(value)

    def _update_progress(self, current: int, total: int):
        self.progress["maximum"] = total
        self.progress["value"] = current
        self.progress.update_idletasks()

    def _update_plot(self):
        self.ax.cla()
        for name, vals in self.returns_by_policy.items():
            light = self.method_colors_light.get(name, "gray")
            bold = self.method_colors_bold.get(name, "black")
            self.ax.plot(vals, color=light, lw=1, label=f"{name} returns")
            if len(vals) >= 5:
                win = 5
                mov = [sum(vals[max(0, i - win + 1):i + 1]) / len(vals[max(0, i - win + 1):i + 1]) for i in range(len(vals))]
                self.ax.plot(mov, color=bold, lw=2, label=f"{name} avg")
        self.ax.set_title("Episode Returns")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Return")
        self.ax.legend()
        self.canvas_plot.draw_idle()

    def _draw_grid(self):
        w = self.grid_canvas.winfo_width() or 500
        h = self.grid_canvas.winfo_height() or 300
        cols = self.env.width
        rows = self.env.height
        cell_w = w / cols
        cell_h = h / rows
        self.grid_canvas.delete("all")
        mat = self.env.to_matrix()
        for y in range(rows):
            for x in range(cols):
                code = mat[y][x]
                x0 = x * cell_w
                y0 = y * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                fill = "white"
                if code == 1:
                    fill = "dim gray"
                elif code == 2:
                    fill = "green"
                elif code == 3:
                    fill = "gold"
                elif code == 4:
                    fill = "sky blue"
                self.grid_canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="black")
                self.grid_canvas.create_text(x0 + 10, y0 + 10, text=f"{x},{y}", fill="light gray", anchor="nw", font=("Arial", 7))
        # path arrows (last episode)
        if len(self.agent.last_trajectory) > 1:
            pts = [step["state"] for step in self.agent.last_trajectory] + [self.agent.last_trajectory[-1]["next_state"]]
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                self.grid_canvas.create_line(
                    x0 * cell_w + cell_w / 2,
                    y0 * cell_h + cell_h / 2,
                    x1 * cell_w + cell_w / 2,
                    y1 * cell_h + cell_h / 2,
                    arrow=tk.LAST,
                    fill="#888888",
                )
        # agent marker
        ax, ay = self.env.state
        self.grid_canvas.create_oval(
            ax * cell_w + cell_w * 0.25,
            ay * cell_h + cell_h * 0.25,
            ax * cell_w + cell_w * 0.75,
            ay * cell_h + cell_h * 0.75,
            fill="blue",
        )
        self.grid_canvas.create_text(ax * cell_w + cell_w / 2, ay * cell_h + cell_h / 2, text="A", fill="white")

    def show_value_table(self):
        self._show_table_dialog("Value Table", value_mode=True)

    def show_q_table(self):
        self._show_table_dialog("Q Table", value_mode=False)

    def _show_table_dialog(self, title: str, value_mode: bool):
        win = tk.Toplevel(self.master)
        win.title(title)
        canvas = tk.Canvas(win, width=500, height=340, bg="white")
        canvas.pack(padx=8, pady=8)
        btn = ttk.Button(win, text="Export trajectories CSV", command=self.export_trajectory)
        btn.pack(pady=4)

        w = 500
        h = 340
        cols = self.env.width
        rows = self.env.height
        cell_w = w / cols
        cell_h = h / rows
        mat = self.env.to_matrix()

        if value_mode:
            values = self.agent.policy.value_table()
        else:
            q = self.agent.policy.q_table()
            values = {s: qv for s, qv in q.items()}

        for y in range(rows):
            for x in range(cols):
                code = mat[y][x]
                x0 = x * cell_w
                y0 = y * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                fill = "white"
                if code == 1:
                    fill = "dim gray"
                elif code == 2:
                    fill = "green"
                elif code == 3:
                    fill = "gold"
                canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="black")
                val = values.get((x, y))
                if value_mode:
                    if val is not None:
                        canvas.create_text(x0 + cell_w / 2, y0 + cell_h / 2, text=f"{val:.2f}")
                else:
                    if val is not None and len(val) >= 4:
                        # action values: 0=Up,1=Down,2=Left,3=Right
                        canvas.create_text(x0 + cell_w / 2, y0 + 10, text=f"U:{val[0]:.2f}", font=("Arial", 7))
                        canvas.create_text(x0 + cell_w / 2, y1 - 10, text=f"D:{val[1]:.2f}", font=("Arial", 7))
                        canvas.create_text(x0 + 12, y0 + cell_h / 2, text=f"L:{val[2]:.2f}", font=("Arial", 7), anchor="w")
                        canvas.create_text(x1 - 12, y0 + cell_h / 2, text=f"R:{val[3]:.2f}", font=("Arial", 7), anchor="e")

    def export_trajectory(self):
        policy_name = self.current_policy_name.replace(" ", "_")
        default_name = f"trajectory_{policy_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=default_name)
        if not file_path:
            return
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "step", "state_x", "state_y", "action", "next_state_x", "next_state_y", "reward", "done"])
            for i, step in enumerate(self.agent.last_trajectory):
                sx, sy = step["state"]
                nx, ny = step["next_state"]
                writer.writerow([self.agent.episode_idx, i, sx, sy, step["action"], nx, ny, step["reward"], step["done"]])

    def save_plot(self):
        policy_name = self.current_policy_name.replace(" ", "_")
        default_name = f"plot_{policy_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_name)
        if not file_path:
            return
        self.fig.savefig(file_path)

    @staticmethod
    def _parse_xy(text: str, default: State) -> State:
        try:
            x_str, y_str = text.split(",")
            return int(x_str), int(y_str)
        except Exception:
            return default

    @staticmethod
    def _parse_blocked(text: str) -> List[State]:
        blocks: List[State] = []
        for part in text.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                x_str, y_str = part.split(",")
                blocks.append((int(x_str), int(y_str)))
            except Exception:
                pass
        return blocks
