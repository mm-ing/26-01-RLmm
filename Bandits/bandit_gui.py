import tkinter as tk
from tkinter import ttk
from typing import List

from bandit_logic import OpenArmedBandit, Agent

# matplotlib is optional; use it if available for plotting cumulative reward
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class BanditGUI:
    def __init__(self, master: tk.Tk, envs: List[OpenArmedBandit], agent: Agent):
        self.master = master
        master.title("3-Armed Bandit")

        # Environments and external agent (injected)
        self.envs = envs
        self.agent = agent

        # Global counters (manual + agent)
        self.total_pulls = 0
        self.total_reward = 0
        self.per_pulls = [0] * len(self.envs)
        self.per_successes = [0] * len(self.envs)
        # plotting series: keep previous runs when switching policies
        self.plot_series = []
        self.active_series_idx = 0
        # display names and colors for policies
        self.policy_options = [
            "Epsilon-Greedy",
            "Thompson Sampling (Bayesian Bandits)",
        ]
        self.policy_colors = {
            "Epsilon-Greedy": "tab:blue",
            "Thompson Sampling (Bayesian Bandits)": "tab:orange",
        }

        # Controls frame
        controls = ttk.Frame(master)
        controls.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        ttk.Label(controls, text="Agent loops (n):").grid(row=0, column=0, sticky="w")
        self.loops_var = tk.IntVar(value=100)
        ttk.Entry(controls, textvariable=self.loops_var, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(controls, text="Agent memory (0=all):").grid(row=1, column=0, sticky="w")
        self.memory_var = tk.IntVar(value=0)
        ttk.Entry(controls, textvariable=self.memory_var, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(controls, text="Epsilon:").grid(row=2, column=0, sticky="w")
        self.epsilon_var = tk.DoubleVar(value=getattr(self.agent, 'epsilon', 0.1))
        ttk.Entry(controls, textvariable=self.epsilon_var, width=8).grid(row=2, column=1, sticky="w")

        ttk.Label(controls, text="Decay:").grid(row=3, column=0, sticky="w")
        self.decay_var = tk.DoubleVar(value=getattr(self.agent, 'decay', 1.0))
        ttk.Entry(controls, textvariable=self.decay_var, width=8).grid(row=3, column=1, sticky="w")

        ttk.Button(controls, text="Recreate Agent", command=self.recreate_agent).grid(row=4, column=0, columnspan=2, pady=4)

        ttk.Label(controls, text="Policy:").grid(row=5, column=0, sticky="w")
        # set initial selection based on injected agent
        init_sel = self.policy_options[0] if getattr(self.agent, 'policy_name', '').lower().startswith('epsilon') else self.policy_options[1]
        self.method_var = tk.StringVar(value=init_sel)
        self.method_combo = ttk.Combobox(controls, values=self.policy_options, textvariable=self.method_var, state="readonly", width=30)
        self.method_combo.grid(row=5, column=1, sticky="w")
        self.method_combo.bind("<<ComboboxSelected>>", self.switch_policy)
        # start first plotting series for initial policy
        init_label = self.policy_options[0] if getattr(self.agent, 'policy_name', '').lower().startswith('epsilon') else self.policy_options[1]
        self._start_new_series(init_label)

        # Manual pull buttons
        manual = ttk.Frame(master)
        manual.grid(row=1, column=0, sticky="nsew", padx=8)
        for i in range(len(self.envs)):
            ttk.Button(manual, text=f"Pull Bandit {i+1}", command=lambda a=i: self.manual_pull(a)).grid(row=0, column=i, padx=4)

        # Agent controls
        agent_frame = ttk.Frame(master)
        agent_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        ttk.Button(agent_frame, text="Agent: single step", command=self.agent_single).grid(row=0, column=0, padx=4)
        ttk.Button(agent_frame, text="Agent: run n loops", command=self.agent_run_n).grid(row=0, column=1, padx=4)
        ttk.Button(agent_frame, text="Reset", command=self.reset_all).grid(row=0, column=2, padx=4)

        # Status / display
        status = ttk.Frame(master)
        status.grid(row=3, column=0, sticky="nsew", padx=8, pady=8)

        self.cumulative_label = ttk.Label(status, text="Cumulative reward: 0", font=("TkDefaultFont", 12, "bold"))
        self.cumulative_label.grid(row=0, column=0, sticky="w")

        self.summary_text = tk.Text(status, width=60, height=8)
        self.summary_text.grid(row=1, column=0)
        self.summary_text.configure(state="disabled")

        # cumulative reward history for plotting
        self.cum_rewards: List[int] = [0]
        if _HAS_MPL:
            # create plot area in the status frame
            self._init_plot(status)

        self._update_summary()

    def recreate_agent(self):
        n = len(self.envs)
        mem = max(0, int(self.memory_var.get()))
        eps = float(self.epsilon_var.get())
        dec = float(self.decay_var.get())
        # recreate agent with currently selected policy (does not reset plot)
        sel = self.method_var.get()
        if sel.startswith("Epsilon"):
            self.agent = Agent(n_arms=n, epsilon=eps, decay=dec, memory=mem, policy_name='epsilon')
        else:
            self.agent = Agent(n_arms=n, epsilon=eps, decay=dec, memory=mem, policy_name='thompson')
        self._update_summary()

    def manual_pull(self, arm: int):
        reward = self.envs[arm].pull()
        self.total_pulls += 1
        self.total_reward += reward
        self.per_pulls[arm] += 1
        self.per_successes[arm] += reward
        # record for plot in the active series
        self.plot_series[self.active_series_idx]['y'].append(self.total_reward)
        self._update_summary()

    def agent_single(self):
        arm, reward = self.agent.run_one(self.envs)
        # update global counters to include this agent action
        self.total_pulls += 1
        self.total_reward += reward
        self.per_pulls[arm] += 1
        self.per_successes[arm] += reward
        # record for plot in the active series
        self.plot_series[self.active_series_idx]['y'].append(self.total_reward)
        self._update_summary()

    def agent_run_n(self):
        n = max(0, int(self.loops_var.get()))
        if n <= 0:
            return
        # run in short batches to keep GUI responsive
        self._agent_run_batch(remaining=n, batch_size=10)

    def _agent_run_batch(self, remaining: int, batch_size: int = 10):
        run_now = min(remaining, batch_size)
        for _ in range(run_now):
            arm, reward = self.agent.run_one(self.envs)
            self.total_pulls += 1
            self.total_reward += reward
            self.per_pulls[arm] += 1
            self.per_successes[arm] += reward
            # record for plot in the active series
            self.plot_series[self.active_series_idx]['y'].append(self.total_reward)
        self._update_summary()
        remaining -= run_now
        if remaining > 0:
            # schedule next batch
            self.master.after(1, lambda: self._agent_run_batch(remaining, batch_size))

    def reset_all(self):
        self.total_pulls = 0
        self.total_reward = 0
        self.per_pulls = [0] * len(self.envs)
        self.per_successes = [0] * len(self.envs)
        self.recreate_agent()
        # clear plotting series and start fresh
        self.plot_series = []
        init_label = self.method_var.get()
        self._start_new_series(init_label)
        self._update_summary()

    def _update_summary(self):
        self.cumulative_label.config(text=f"Cumulative reward: {self.total_reward}")
        lines = [f"Total pulls: {self.total_pulls}", ""]
        for i in range(len(self.envs)):
            pulls = self.per_pulls[i]
            succ = self.per_successes[i]
            rate = f"{succ/pulls:.3f}" if pulls > 0 else "N/A"
            lines.append(f"Bandit {i+1}: pulls={pulls} successes={succ} success_rate={rate}")

        lines.append("")
        eps_val = getattr(getattr(self.agent, 'policy', None), 'epsilon', None)
        lines.append(f"Agent epsilon={eps_val:.5f}" if eps_val is not None else "Agent epsilon=N/A")
        lines.append(f"Policy: {self.method_var.get()}")

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")
        # update plot if available
        if _HAS_MPL:
            self._update_plot()

    def _init_plot(self, parent_frame: ttk.Frame):
        self.figure = Figure(figsize=(4, 2.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Cumulative Reward")
        self.ax.set_xlabel("Pulls")
        self.ax.set_ylabel("Cumulative Reward")
        self.ax.grid(True)
        # initial empty plot; series are drawn in _update_plot
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent_frame)
        self.canvas.get_tk_widget().grid(row=1, column=1, padx=8)

    def _update_plot(self):
        if not hasattr(self, 'ax'):
            return
        self.ax.cla()
        self.ax.set_title("Cumulative Reward")
        self.ax.set_xlabel("Pulls")
        self.ax.set_ylabel("Cumulative Reward")
        for series in self.plot_series:
            y = series['y']
            x = list(range(len(y)))
            self.ax.plot(x, y, lw=2, label=series.get('label'), color=series.get('color'))
        self.ax.grid(True)
        if len(self.plot_series) > 0:
            self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _start_new_series(self, display_label: str):
        color = self.policy_colors.get(display_label, 'tab:blue')
        self.plot_series.append({'label': display_label, 'color': color, 'y': [0]})
        self.active_series_idx = len(self.plot_series) - 1

    def switch_policy(self, *_):
        # called when user selects a new policy from combobox; reset agent and counters
        sel = self.method_var.get()
        n = len(self.envs)
        mem = max(0, int(self.memory_var.get()))
        eps = float(self.epsilon_var.get())
        dec = float(self.decay_var.get())
        # recreate agent using chosen policy and reset counters
        if sel.startswith("Epsilon"):
            self.agent = Agent(n_arms=n, epsilon=eps, decay=dec, memory=mem, policy_name='epsilon')
        else:
            self.agent = Agent(n_arms=n, epsilon=eps, decay=dec, memory=mem, policy_name='thompson')
        # reset counts but keep existing plotted series intact
        self.total_pulls = 0
        self.total_reward = 0
        self.per_pulls = [0] * n
        self.per_successes = [0] * n
        # start a new plotting series for this policy
        self._start_new_series(sel)
        self._update_summary()


def main(envs: List[OpenArmedBandit], agent: Agent):
    root = tk.Tk()
    app = BanditGUI(root, envs=envs, agent=agent)
    root.mainloop()


if __name__ == "__main__":
    # fallback: create default envs and agent when run directly
    default_probs = [0.2, 0.5, 0.8]
    envs = [OpenArmedBandit(p) for p in default_probs]
    agent = Agent(n_arms=len(envs))
    main(envs, agent)
