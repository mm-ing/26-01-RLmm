"""
HalfCheetah GUI with Tkinter
Provides interactive interface for training and visualizing RL algorithms
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.figure import Figure
import time
from PIL import Image, ImageTk
import cv2


class HalfCheetahGUI:
    """GUI for HalfCheetah RL training and visualization"""
    
    def __init__(self, root, environment, agent, policies):
        self.root = root
        self.environment = environment
        self.agent = agent
        self.policies = policies
        
        self.root.title("HalfCheetah RL Training")
        self.root.geometry("1400x900")
        
        # Dark mode colors
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
        
        # Statistics
        self.method_stats = {}
        for method_name in self.policies.keys():
            self.method_stats[method_name] = {
                'returns': [],
                'color': self._get_color(method_name)
            }
        
        # Animation
        self.show_animation = tk.BooleanVar(value=False)
        self.current_frame = None
        self.animation_canvas = None
        self.animation_image_id = None
        
        self._create_widgets()
        self._setup_plot()
        
    def _get_color(self, method_name):
        """Get color for method"""
        colors = {
            'PPO': '#f7b731',
            'TD3': '#4ecdc4',
            'SAC': '#5f27cd'
        }
        return colors.get(method_name, '#95afc0')
    
    def _create_widgets(self):
        """Create GUI widgets"""
        # Main container with 3 columns
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Animation and controls
        left_panel = tk.Frame(main_frame, bg=self.bg_color)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5)
        
        # Middle panel - Parameters
        middle_panel = tk.Frame(main_frame, bg=self.bg_color)
        middle_panel.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Right panel - Plot
        right_panel = tk.Frame(main_frame, bg=self.bg_color)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=5)
        
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # === LEFT PANEL ===
        # Animation
        anim_frame = tk.LabelFrame(left_panel, text="Environment Animation", 
                                   bg=self.bg_color, fg=self.fg_color)
        anim_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create canvas for animation
        self.animation_canvas = tk.Canvas(anim_frame, width=400, height=300, 
                                         bg=self.bg_color, highlightthickness=0)
        self.animation_canvas.pack(pady=10)
        
        tk.Checkbutton(anim_frame, text="Enable Animation", 
                      variable=self.show_animation,
                      bg=self.bg_color, fg=self.fg_color,
                      selectcolor=self.entry_bg,
                      command=self._toggle_animation).pack(pady=5)
        
        # Environment parameters
        env_frame = tk.LabelFrame(left_panel, text="Environment Parameters",
                                 bg=self.bg_color, fg=self.fg_color)
        env_frame.pack(fill=tk.X, pady=5)
        
        # Environment parameter grid
        env_grid = tk.Frame(env_frame, bg=self.bg_color)
        env_grid.pack(fill=tk.X, padx=5, pady=5)
        
        self.env_param_vars = {}
        env_params = [
            ('Forward Reward Weight', 'forward_reward_weight', '1.0'),
            ('Ctrl Cost Weight', 'ctrl_cost_weight', '0.1'),
            ('Reset Noise Scale', 'reset_noise_scale', '0.1')
        ]
        
        for idx, (label, key, default) in enumerate(env_params):
            row = idx // 3
            col = idx % 3
            
            tk.Label(env_grid, text=label + ":", bg=self.bg_color, 
                    fg=self.fg_color, font=('Arial', 8)).grid(row=row*2, column=col, sticky='w', padx=2, pady=2)
            
            var = tk.StringVar(value=default)
            self.env_param_vars[key] = var
            tk.Entry(env_grid, textvariable=var, bg=self.entry_bg, 
                    fg=self.fg_color, width=10).grid(row=row*2+1, column=col, padx=2, pady=2)
        
        self.exclude_pos_var = tk.BooleanVar(value=True)
        tk.Checkbutton(env_frame, text="Exclude Current Positions", 
                      variable=self.exclude_pos_var,
                      bg=self.bg_color, fg=self.fg_color,
                      selectcolor=self.entry_bg).pack(pady=5)
        
        tk.Button(env_frame, text="Update Environment", command=self._update_environment,
                 bg=self.button_bg, fg=self.fg_color).pack(pady=5, fill=tk.X, padx=5)
        
        # Method selection
        method_frame = tk.LabelFrame(left_panel, text="Method Selection",
                                    bg=self.bg_color, fg=self.fg_color)
        method_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(method_frame, text="Algorithm:", bg=self.bg_color, 
                fg=self.fg_color).pack(pady=2)
        
        self.method_var = tk.StringVar(value=list(self.policies.keys())[0])
        method_dropdown = ttk.Combobox(method_frame, textvariable=self.method_var,
                                      values=list(self.policies.keys()),
                                      state='readonly')
        method_dropdown.pack(pady=2, padx=5, fill=tk.X)
        method_dropdown.bind('<<ComboboxSelected>>', self._on_method_change)
        
        self.compare_var = tk.BooleanVar(value=False)
        tk.Checkbutton(method_frame, text="Compare Mode", 
                      variable=self.compare_var,
                      bg=self.bg_color, fg=self.fg_color,
                      selectcolor=self.entry_bg,
                      command=self._toggle_compare_mode).pack(pady=5)
        
        # Compare methods selection
        self.compare_frame = tk.LabelFrame(left_panel, text="Compare Methods",
                                          bg=self.bg_color, fg=self.fg_color)
        self.compare_frame.pack(fill=tk.X, pady=5)
        self.compare_frame.pack_forget()
        
        self.compare_vars = {}
        for method_name in self.policies.keys():
            var = tk.BooleanVar(value=False)
            self.compare_vars[method_name] = var
            tk.Checkbutton(self.compare_frame, text=method_name, variable=var,
                          bg=self.bg_color, fg=self.fg_color,
                          selectcolor=self.entry_bg).pack(anchor='w', padx=5)
        
        # Control buttons
        control_frame = tk.LabelFrame(left_panel, text="Training Control",
                                     bg=self.bg_color, fg=self.fg_color)
        control_frame.pack(fill=tk.X, pady=5)
        
        btn_frame = tk.Frame(control_frame, bg=self.bg_color)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Reset", command=self._reset_training,
                 bg=self.button_bg, fg=self.fg_color, width=10).grid(row=0, column=0, padx=2)
        tk.Button(btn_frame, text="Train", command=self._start_training,
                 bg=self.button_bg, fg=self.fg_color, width=10).grid(row=0, column=1, padx=2)
        tk.Button(btn_frame, text="Run", command=self._run_episode,
                 bg=self.button_bg, fg=self.fg_color, width=10).grid(row=0, column=2, padx=2)
        
        tk.Button(control_frame, text="Stop Training", command=self._stop_training,
                 bg=self.button_bg, fg=self.fg_color).pack(pady=2, fill=tk.X, padx=5)
        tk.Button(control_frame, text="Save Plot", command=self._save_plot,
                 bg=self.button_bg, fg=self.fg_color).pack(pady=2, fill=tk.X, padx=5)
        
        # === MIDDLE PANEL ===
        # Common parameters
        common_frame = tk.LabelFrame(middle_panel, text="Common Parameters",
                                    bg=self.bg_color, fg=self.fg_color)
        common_frame.pack(fill=tk.X, pady=5)
        
        params_grid = tk.Frame(common_frame, bg=self.bg_color)
        params_grid.pack(fill=tk.X, padx=5, pady=5)
        
        self.param_vars = {}
        common_params = [
            ('Episodes', 'episodes', '5000'),
            ('Learning Rate', 'lr', '0.0003'),
            ('Gamma', 'gamma', '0.99'),
            ('Hidden Dims', 'hidden_dims', '256,256'),
            ('Activation', 'activation', 'relu')
        ]
        
        for idx, (label, key, default) in enumerate(common_params):
            row = idx // 3
            col = idx % 3
            
            tk.Label(params_grid, text=label + ":", bg=self.bg_color, 
                    fg=self.fg_color).grid(row=row*2, column=col, sticky='w', padx=2, pady=2)
            
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            tk.Entry(params_grid, textvariable=var, bg=self.entry_bg, 
                    fg=self.fg_color, width=12).grid(row=row*2+1, column=col, padx=2, pady=2)
        
        # Method-specific parameters with scrollbar
        self.method_params_frame = tk.LabelFrame(middle_panel, text="Method Parameters",
                                                bg=self.bg_color, fg=self.fg_color)
        self.method_params_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add canvas and scrollbar for parameters
        self.params_canvas = tk.Canvas(self.method_params_frame, bg=self.bg_color, highlightthickness=0)
        self.params_scrollbar = tk.Scrollbar(self.method_params_frame, orient="vertical", command=self.params_canvas.yview)
        self.params_scrollable_frame = tk.Frame(self.params_canvas, bg=self.bg_color)
        
        self.params_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
        )
        
        self.params_canvas.create_window((0, 0), window=self.params_scrollable_frame, anchor="nw")
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
        
        self.params_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.params_scrollbar.pack(side="right", fill="y")
        
        self._create_method_params()
        
        # Status
        status_frame = tk.LabelFrame(middle_panel, text="Status",
                                    bg=self.bg_color, fg=self.fg_color)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, bg=self.entry_bg,
                                   fg=self.fg_color, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === RIGHT PANEL ===
        plot_frame = tk.LabelFrame(right_panel, text="Reward Plot",
                                  bg=self.bg_color, fg=self.fg_color)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.plot_canvas_frame = tk.Frame(plot_frame, bg=self.bg_color)
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def _create_method_params(self):
        """Create method-specific parameter inputs"""
        # Clear existing
        for widget in self.params_scrollable_frame.winfo_children():
            widget.destroy()
        
        method = self.method_var.get()
        
        params_grid = tk.Frame(self.params_scrollable_frame, bg=self.bg_color)
        params_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        method_specific_params = {
            'PPO': [
                ('Clip Epsilon', 'clip_epsilon', '0.2'),
                ('Value Coef', 'value_coef', '0.5'),
                ('Entropy Coef', 'entropy_coef', '0.01'),
                ('Update Epochs', 'update_epochs', '10'),
                ('Network Activation', 'network_activation', 'relu')
            ],
            'TD3': [
                ('Tau', 'tau', '0.005'),
                ('Policy Noise', 'policy_noise', '0.2'),
                ('Noise Clip', 'noise_clip', '0.5'),
                ('Policy Delay', 'policy_delay', '2'),
                ('Buffer Size', 'buffer_size', '100000'),
                ('Batch Size', 'batch_size', '64'),
                ('Exploration Noise', 'exploration_noise', '0.1'),
                ('Actor Activation', 'actor_activation', 'relu'),
                ('Critic Activation', 'critic_activation', 'relu')
            ],
            'SAC': [
                ('Tau', 'tau', '0.005'),
                ('Alpha', 'alpha', '0.2'),
                ('Buffer Size', 'buffer_size', '100000'),
                ('Batch Size', 'batch_size', '64'),
                ('Train Freq', 'train_freq', '4'),
                ('Gradient Steps', 'gradient_steps', '1'),
                ('Actor Activation', 'actor_activation', 'relu'),
                ('Critic Activation', 'critic_activation', 'relu')
            ]
        }
        
        params = method_specific_params.get(method, [])
        
        for idx, (label, key, default) in enumerate(params):
            row = idx // 3
            col = idx % 3
            
            tk.Label(params_grid, text=label + ":", bg=self.bg_color,
                    fg=self.fg_color).grid(row=row*2, column=col, sticky='w', padx=2, pady=2)
            
            param_key = f"{method}_{key}"
            if param_key not in self.param_vars:
                self.param_vars[param_key] = tk.StringVar(value=default)
            
            tk.Entry(params_grid, textvariable=self.param_vars[param_key],
                    bg=self.entry_bg, fg=self.fg_color, 
                    width=12).grid(row=row*2+1, column=col, padx=2, pady=2)
    
    def _setup_plot(self):
        """Setup matplotlib plot"""
        self.fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, facecolor='#2b2b2b')
        
        self.ax.set_xlabel('Episode', color='white')
        self.ax.set_ylabel('Reward', color='white')
        self.ax.set_title('Training Progress', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _toggle_animation(self):
        """Toggle animation on/off"""
        if self.show_animation.get():
            # Set environment to rgb_array render mode
            self.environment.render_mode = 'rgb_array'
            self.environment.reset_env()
            self._log_status("Animation enabled")
        else:
            # Disable rendering
            self.environment.render_mode = None
            self.environment.reset_env()
            # Clear canvas
            if self.animation_canvas:
                self.animation_canvas.delete("all")
            self._log_status("Animation disabled")
    
    def _update_animation_frame(self, frame):
        """Update animation canvas with new frame"""
        if frame is None or not self.show_animation.get():
            return
        
        try:
            # MuJoCo/Gymnasium renders in RGB format already (not BGR)
            frame_rgb = frame
            
            # Resize frame to fit canvas
            canvas_width = 400
            canvas_height = 300
            h, w = frame_rgb.shape[:2]
            scale = min(canvas_width/w, canvas_height/h)
            new_w, new_h = int(w*scale), int(h*scale)
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            # Convert to PIL Image (frame is already in RGB format)
            img = Image.fromarray(frame_resized.astype('uint8'))
            photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            def update_canvas():
                if self.animation_image_id:
                    self.animation_canvas.delete(self.animation_image_id)
                
                x = (canvas_width - new_w) // 2
                y = (canvas_height - new_h) // 2
                self.animation_image_id = self.animation_canvas.create_image(
                    x, y, anchor=tk.NW, image=photo
                )
                self.animation_canvas.image = photo  # Keep reference
            
            self.root.after(0, update_canvas)
        except Exception as e:
            # Log error for debugging
            self._log_status(f"Animation error: {str(e)}")
    
    def _update_environment(self):
        """Update environment parameters"""
        try:
            self.environment.forward_reward_weight = float(self.env_param_vars['forward_reward_weight'].get())
            self.environment.ctrl_cost_weight = float(self.env_param_vars['ctrl_cost_weight'].get())
            self.environment.reset_noise_scale = float(self.env_param_vars['reset_noise_scale'].get())
            self.environment.exclude_current_positions_from_observation = self.exclude_pos_var.get()
            
            # Maintain render mode if animation is enabled
            if self.show_animation.get():
                self.environment.render_mode = 'rgb_array'
            
            self.environment.reset_env()
            self._log_status(f"Environment updated: FRW={self.environment.forward_reward_weight}, "
                           f"CCW={self.environment.ctrl_cost_weight}, "
                           f"RNS={self.environment.reset_noise_scale}")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
    
    def _on_method_change(self, event=None):
        """Handle method selection change"""
        self._create_method_params()
    
    def _toggle_compare_mode(self):
        """Toggle compare mode"""
        if self.compare_var.get():
            self.compare_frame.pack(fill=tk.X, pady=5, after=self.compare_frame.master.winfo_children()[2])
            self.compare_mode = True
        else:
            self.compare_frame.pack_forget()
            self.compare_mode = False
    
    def _get_method_params(self, method_name):
        """Get parameters for method"""
        params = {
            'state_dim': self.environment.get_state_dim(),
            'action_dim': self.environment.get_action_dim(),
            'lr': float(self.param_vars['lr'].get()),
            'gamma': float(self.param_vars['gamma'].get()),
            'hidden_dims': [int(x.strip()) for x in self.param_vars['hidden_dims'].get().split(',')]
        }
        
        # Method-specific parameters
        if method_name == 'PPO':
            params.update({
                'clip_epsilon': float(self.param_vars.get('PPO_clip_epsilon', tk.StringVar(value='0.2')).get()),
                'value_coef': float(self.param_vars.get('PPO_value_coef', tk.StringVar(value='0.5')).get()),
                'entropy_coef': float(self.param_vars.get('PPO_entropy_coef', tk.StringVar(value='0.01')).get()),
                'update_epochs': int(self.param_vars.get('PPO_update_epochs', tk.StringVar(value='10')).get())
            })
        elif method_name == 'TD3':
            params.update({
                'tau': float(self.param_vars.get('TD3_tau', tk.StringVar(value='0.005')).get()),
                'policy_noise': float(self.param_vars.get('TD3_policy_noise', tk.StringVar(value='0.2')).get()),
                'noise_clip': float(self.param_vars.get('TD3_noise_clip', tk.StringVar(value='0.5')).get()),
                'policy_delay': int(self.param_vars.get('TD3_policy_delay', tk.StringVar(value='2')).get()),
                'buffer_size': int(self.param_vars.get('TD3_buffer_size', tk.StringVar(value='100000')).get()),
                'batch_size': int(self.param_vars.get('TD3_batch_size', tk.StringVar(value='64')).get()),
                'exploration_noise': float(self.param_vars.get('TD3_exploration_noise', tk.StringVar(value='0.1')).get()),
                'actor_activation': self.param_vars.get('TD3_actor_activation', tk.StringVar(value='relu')).get(),
                'critic_activation': self.param_vars.get('TD3_critic_activation', tk.StringVar(value='relu')).get()
            })
        elif method_name == 'SAC':
            params.update({
                'tau': float(self.param_vars.get('SAC_tau', tk.StringVar(value='0.005')).get()),
                'alpha': float(self.param_vars.get('SAC_alpha', tk.StringVar(value='0.2')).get()),
                'buffer_size': int(self.param_vars.get('SAC_buffer_size', tk.StringVar(value='100000')).get()),
                'batch_size': int(self.param_vars.get('SAC_batch_size', tk.StringVar(value='64')).get()),
                'train_freq': int(self.param_vars.get('SAC_train_freq', tk.StringVar(value='4')).get()),
                'gradient_steps': int(self.param_vars.get('SAC_gradient_steps', tk.StringVar(value='1')).get()),
                'actor_activation': self.param_vars.get('SAC_actor_activation', tk.StringVar(value='relu')).get(),
                'critic_activation': self.param_vars.get('SAC_critic_activation', tk.StringVar(value='relu')).get()
            })
        
        return params
    
    def _reset_training(self):
        """Reset training"""
        self._stop_training()
        
        for method_name in self.method_stats.keys():
            self.method_stats[method_name]['returns'] = []
        
        self.agent.reset_stats()
        self._update_plot()
        self._log_status("Training reset")
    
    def _start_training(self):
        """Start training"""
        if self.training:
            messagebox.showwarning("Warning", "Training already in progress")
            return
        
        self.training = True
        
        if self.compare_mode:
            selected_methods = [name for name, var in self.compare_vars.items() if var.get()]
            if not selected_methods:
                messagebox.showwarning("Warning", "No methods selected for comparison")
                self.training = False
                return
            
            for method_name in selected_methods:
                thread = threading.Thread(target=self._train_method, args=(method_name,))
                thread.daemon = True
                self.training_threads[method_name] = thread
                thread.start()
        else:
            method_name = self.method_var.get()
            thread = threading.Thread(target=self._train_method, args=(method_name,))
            thread.daemon = True
            self.training_threads[method_name] = thread
            thread.start()
    
    def _train_method(self, method_name):
        """Train a specific method"""
        try:
            episodes = int(self.param_vars['episodes'].get())
            
            # Create fresh policy instance
            params = self._get_method_params(method_name)
            policy_class = self.policies[method_name]
            policy = policy_class(**params)
            
            # Reset agent's internal stats for clean tracking
            self.agent.episode_rewards = []
            
            self._log_status(f"Training {method_name} for {episodes} episodes...")
            
            for episode in range(episodes):
                if not self.training or method_name in self.paused_methods:
                    break
                
                # Set current policy
                self.agent.set_policy(policy)
                
                # Run episode with rendering if enabled and this is the latest method
                render = self.show_animation.get()
                render_callback = self._update_animation_frame if render else None
                reward = self.agent.run_episode(render=render, train=True, 
                                                render_callback=render_callback)
                
                # Update stats
                self.method_stats[method_name]['returns'].append(reward)
                
                # Update plot more frequently for early episodes
                if episode < 20 or episode % 5 == 0 or episode == episodes - 1:
                    self.root.after(0, self._update_plot)
                
                # Log progress
                if episode < 5 or episode % 5 == 0:
                    method_returns = self.method_stats[method_name]['returns']
                    avg_reward = np.mean(method_returns[-10:]) if len(method_returns) >= 10 else np.mean(method_returns) if method_returns else reward
                    memory_size = len(policy.memory) if hasattr(policy, 'memory') else 'N/A'
                    self._log_status(f"{method_name} - Episode {episode+1}/{episodes}, Reward: {reward:.2f}, Avg: {avg_reward:.2f}, Memory: {memory_size}")
            
            self._log_status(f"{method_name} training completed!")
            
        except Exception as e:
            self._log_status(f"Error training {method_name}: {str(e)}")
        finally:
            if method_name in self.training_threads:
                del self.training_threads[method_name]
            if len(self.training_threads) == 0:
                self.training = False
    
    def _run_episode(self):
        """Run a single episode without training"""
        method_name = self.method_var.get()
        params = self._get_method_params(method_name)
        policy_class = self.policies[method_name]
        policy = policy_class(**params)
        
        self.agent.set_policy(policy)
        
        thread = threading.Thread(target=self._run_episode_thread)
        thread.daemon = True
        thread.start()
    
    def _run_episode_thread(self):
        """Run episode in thread"""
        try:
            render = self.show_animation.get()
            render_callback = self._update_animation_frame if render else None
            reward = self.agent.run_episode(render=render, train=False,
                                          render_callback=render_callback)
            self._log_status(f"Episode reward: {reward:.2f}")
        except Exception as e:
            self._log_status(f"Error running episode: {str(e)}")
    
    def _stop_training(self):
        """Stop training"""
        self.training = False
        self.training_threads.clear()
        self._log_status("Training stopped")
    
    def _update_plot(self):
        """Update reward plot"""
        self.ax.clear()
        
        for method_name, stats in self.method_stats.items():
            if len(stats['returns']) == 0:
                continue
            
            returns = stats['returns']
            episodes = list(range(1, len(returns) + 1))
            color = stats['color']
            
            # Plot individual rewards (light color)
            self.ax.plot(episodes, returns, color=color, alpha=0.3, 
                        linewidth=0.5, zorder=1, label=f"{method_name} (raw)")
            
            # Plot moving average (bold color)
            if len(returns) >= 10:
                window = min(20, len(returns))
                moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
                avg_episodes = list(range(window, len(returns) + 1))
                self.ax.plot(avg_episodes, moving_avg, color=color, 
                           linewidth=2, zorder=2, label=f"{method_name} (avg)")
        
        self.ax.set_xlabel('Episode', color='white')
        self.ax.set_ylabel('Reward', color='white')
        self.ax.set_title('Training Progress', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.legend(loc='lower left', facecolor='#2b2b2b', edgecolor='white', 
                      labelcolor='white')
        self.ax.grid(True, alpha=0.2, color='white')
        
        self.canvas.draw()
    
    def _save_plot(self):
        """Save plot to image"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"half_cheetah_plot_{timestamp}.png"
        self.fig.savefig(filename, facecolor='#2b2b2b', edgecolor='white')
        self._log_status(f"Plot saved to {filename}")
        messagebox.showinfo("Success", f"Plot saved to {filename}")
    
    def _log_status(self, message):
        """Log message to status text"""
        def update():
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.see(tk.END)
        
        if threading.current_thread() == threading.main_thread():
            update()
        else:
            self.root.after(0, update)
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()
