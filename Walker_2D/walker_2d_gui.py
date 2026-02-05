"""Walker2D GUI - Tkinter interface for RL training visualization"""
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
from PIL import Image, ImageTk
import logging
from walker_2d_logic import PPO, TD3, SAC, Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Walker2DGUI:
    """Tkinter GUI for Walker2D RL training"""
    
    def __init__(self, root, environment, agent, policies):
        """
        Initialize GUI
        
        Args:
            root: Tkinter root window
            environment: Walker2D environment
            agent: Agent instance
            policies: Dictionary of available policies
        """
        self.root = root
        self.env = environment
        self.agent = agent
        self.policies = policies
        
        self.root.title("Walker2D RL Workbench")
        self.root.configure(bg='#2b2b2b')
        
        # Training state
        self.training = False
        self.training_threads = []
        self.stop_training = False
        self.animation_enabled = False
        self.render_thread_id = None  # Track which thread handles rendering
        
        # Plotting data
        self.rewards_data = {}
        self.episode_counters = {}
        
        # Method colors
        self.colors = {
            'PPO': '#FF6B6B',
            'TD3': '#4ECDC4',
            'SAC': '#45B7D1'
        }
        
        # Extended color palette for multiple configurations
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D', '#6BCB77',
            '#FF9A76', '#A29BFE', '#FD79A8', '#FDCB6E', '#74B9FF',
            '#55EFC4', '#DFE6E9', '#FF7675', '#00B894', '#FFEAA7'
        ]
        self.config_colors = {}  # Map config names to colors
        
        # TD3 range mode variables
        self.td3_range_mode = {}
        self.td3_noise_vars = {}
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top: Animation
        self._create_animation_section(main_frame)
        
        # Middle: Controls
        control_frame = tk.Frame(main_frame, bg='#2b2b2b')
        control_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Environment and Method params
        left_frame = tk.Frame(control_frame, bg='#2b2b2b')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self._create_environment_params(left_frame)
        self._create_method_selection(left_frame)
        self._create_method_params(left_frame)
        
        # Right: Neural network params and controls
        right_frame = tk.Frame(control_frame, bg='#2b2b2b')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self._create_network_params(right_frame)
        self._create_control_buttons(right_frame)
        
        # Bottom: Plot
        self._create_plot_section(main_frame)
        
    def _create_animation_section(self, parent):
        """Create animation canvas section"""
        anim_frame = tk.LabelFrame(parent, text="Environment Animation", 
                                   bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
        anim_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Animation canvas
        self.anim_canvas = tk.Canvas(anim_frame, width=360, height=360, bg='black')
        self.anim_canvas.pack(pady=10)
        
        # Animation control
        btn_frame = tk.Frame(anim_frame, bg='#2b2b2b')
        btn_frame.pack(pady=5)
        
        self.anim_var = tk.BooleanVar(value=False)
        anim_check = tk.Checkbutton(btn_frame, text="Enable Animation", variable=self.anim_var,
                                   command=self._toggle_animation, bg='#2b2b2b', fg='white',
                                   selectcolor='#404040', font=('Arial', 9))
        anim_check.pack()
        
    def _create_environment_params(self, parent):
        """Create environment parameter controls"""
        env_frame = tk.LabelFrame(parent, text="Environment Parameters",
                                 bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
        env_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.env_params = {}
        
        params = [
            ('forward_reward_weight', 1.0, 0.0, 5.0),
            ('ctrl_cost_weight', 0.001, 0.0, 0.1),
            ('healthy_reward', 1.0, 0.0, 10.0),
            ('reset_noise_scale', 0.005, 0.0, 0.1)
        ]
        
        for i, (name, default, min_val, max_val) in enumerate(params):
            label = tk.Label(env_frame, text=name.replace('_', ' ').title() + ':',
                           bg='#2b2b2b', fg='white', font=('Arial', 9))
            label.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            
            var = tk.DoubleVar(value=default)
            entry = tk.Entry(env_frame, textvariable=var, width=10, bg='#404040',
                           fg='white', font=('Arial', 9))
            entry.grid(row=i, column=1, padx=5, pady=2)
            
            self.env_params[name] = var
        
        # Boolean params
        bool_params = [
            ('terminate_when_unhealthy', True)
        ]
        
        i = len(params)
        for name, default in bool_params:
            var = tk.BooleanVar(value=default)
            check = tk.Checkbutton(env_frame, text=name.replace('_', ' ').title(),
                                 variable=var, bg='#2b2b2b', fg='white',
                                 selectcolor='#404040', font=('Arial', 9))
            check.grid(row=i, column=0, columnspan=2, sticky='w', padx=5, pady=2)
            self.env_params[name] = var
            i += 1
        
        # Range params
        range_params = [
            ('healthy_z_range', (0.8, 2.0)),
            ('healthy_angle_range', (-1.0, 1.0))
        ]
        
        for name, (min_default, max_default) in range_params:
            label = tk.Label(env_frame, text=name.replace('_', ' ').title() + ':',
                           bg='#2b2b2b', fg='white', font=('Arial', 9))
            label.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            
            range_frame = tk.Frame(env_frame, bg='#2b2b2b')
            range_frame.grid(row=i, column=1, padx=5, pady=2)
            
            min_var = tk.DoubleVar(value=min_default)
            max_var = tk.DoubleVar(value=max_default)
            
            tk.Entry(range_frame, textvariable=min_var, width=5, bg='#404040',
                   fg='white', font=('Arial', 9)).pack(side=tk.LEFT)
            tk.Label(range_frame, text='-', bg='#2b2b2b', fg='white').pack(side=tk.LEFT, padx=2)
            tk.Entry(range_frame, textvariable=max_var, width=5, bg='#404040',
                   fg='white', font=('Arial', 9)).pack(side=tk.LEFT)
            
            self.env_params[name] = (min_var, max_var)
            i += 1
    
    def _create_method_selection(self, parent):
        """Create method selection controls"""
        method_frame = tk.LabelFrame(parent, text="Method Selection",
                                    bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Method dropdown
        tk.Label(method_frame, text="Method:", bg='#2b2b2b', fg='white',
               font=('Arial', 9)).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.method_var = tk.StringVar(value='PPO')
        method_combo = ttk.Combobox(method_frame, textvariable=self.method_var,
                                   values=['PPO', 'TD3', 'SAC'], state='readonly', width=15)
        method_combo.grid(row=0, column=1, padx=5, pady=5)
        method_combo.bind('<<ComboboxSelected>>', self._on_method_change)
        
        # Compare mode
        self.compare_var = tk.BooleanVar(value=False)
        compare_check = tk.Checkbutton(method_frame, text="Compare Mode",
                                      variable=self.compare_var, bg='#2b2b2b',
                                      fg='white', selectcolor='#404040',
                                      font=('Arial', 9), command=self._on_compare_change)
        compare_check.grid(row=0, column=2, padx=5, pady=5)
        
    def _create_method_params(self, parent):
        """Create method parameter controls"""
        # Create scrollable frame
        params_container = tk.LabelFrame(parent, text="Method Parameters",
                                        bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
        params_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas and scrollbar
        self.params_canvas = tk.Canvas(params_container, bg='#2b2b2b', highlightthickness=0)
        self.params_scrollbar = tk.Scrollbar(params_container, orient="vertical",
                                            command=self.params_canvas.yview)
        self.params_scrollable_frame = tk.Frame(self.params_canvas, bg='#2b2b2b')
        
        self.params_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
        )
        
        self.params_canvas.create_window((0, 0), window=self.params_scrollable_frame, anchor="nw")
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
        
        self.params_canvas.pack(side="left", fill="both", expand=True)
        self.params_scrollbar.pack(side="right", fill="y")
        
        self.method_params = {}
        self._update_method_params()
        
    def _create_network_params(self, parent):
        """Create neural network parameter controls"""
        net_frame = tk.LabelFrame(parent, text="Neural Network Parameters",
                                 bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
        net_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.network_params = {}
        
        # Common network params
        params = [
            ('hidden_layer_1', 256, 32, 512),
            ('hidden_layer_2', 256, 32, 512),
            ('learning_rate', 0.0003, 0.00001, 0.01)
        ]
        
        for i, (name, default, min_val, max_val) in enumerate(params):
            label = tk.Label(net_frame, text=name.replace('_', ' ').title() + ':',
                           bg='#2b2b2b', fg='white', font=('Arial', 9))
            label.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            
            var = tk.DoubleVar(value=default)
            entry = tk.Entry(net_frame, textvariable=var, width=10, bg='#404040',
                           fg='white', font=('Arial', 9))
            entry.grid(row=i, column=1, padx=5, pady=2)
            
            self.network_params[name] = var
        
        # Activation function
        label = tk.Label(net_frame, text="Activation:", bg='#2b2b2b', fg='white',
                       font=('Arial', 9))
        label.grid(row=len(params), column=0, sticky='w', padx=5, pady=2)
        
        self.activation_var = tk.StringVar(value='relu')
        activation_combo = ttk.Combobox(net_frame, textvariable=self.activation_var,
                                       values=['relu', 'tanh', 'leaky_relu'],
                                       state='readonly', width=12)
        activation_combo.grid(row=len(params), column=1, padx=5, pady=2)
        
    def _create_control_buttons(self, parent):
        """Create control buttons"""
        btn_frame = tk.LabelFrame(parent, text="Controls", bg='#2b2b2b',
                                 fg='white', font=('Arial', 10, 'bold'))
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Episodes
        tk.Label(btn_frame, text="Episodes:", bg='#2b2b2b', fg='white',
               font=('Arial', 9)).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.episodes_var = tk.IntVar(value=5000)
        tk.Entry(btn_frame, textvariable=self.episodes_var, width=10, bg='#404040',
               fg='white', font=('Arial', 9)).grid(row=0, column=1, padx=5, pady=5)
        
        # Buttons
        button_configs = [
            ("Reset", self._reset_environment, 1),
            ("Train", self._start_training, 2),
            ("Cancel", self._stop_training_callback, 3),
            ("Save Plot", self._save_plot, 4)
        ]
        
        for text, command, row in button_configs:
            btn = tk.Button(btn_frame, text=text, command=command, bg='#404040',
                          fg='white', font=('Arial', 9, 'bold'), width=12)
            btn.grid(row=row, column=0, columnspan=2, pady=5)
    
    def _create_plot_section(self, parent):
        """Create matplotlib plot section"""
        plot_frame = tk.LabelFrame(parent, text="Reward Plot", bg='#2b2b2b',
                                  fg='white', font=('Arial', 10, 'bold'))
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure
        self.fig = Figure(figsize=(10, 6), facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_xlabel('Episode', color='white')
        self.ax.set_ylabel('Reward', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('#2b2b2b')
        self.ax.spines['right'].set_color('#2b2b2b')
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _get_method_params(self):
        """Get method-specific parameters"""
        method = self.method_var.get()
        
        common_params = [
            ('gamma', 0.99, 0.9, 0.999),
            ('batch_size', 256, 32, 512),
            ('episodes', 5000, 100, 10000)
        ]
        
        if method == 'PPO':
            specific_params = [
                ('clip_ratio', 0.2, 0.1, 0.3),
                ('epochs', 10, 5, 20),
                ('ppo_lr', 0.0003, 0.00001, 0.01)
            ]
        elif method == 'TD3':
            specific_params = [
                ('actor_lr', 0.0003, 0.00001, 0.01),
                ('critic_lr', 0.0003, 0.00001, 0.01),
                ('tau', 0.005, 0.001, 0.01),
                ('policy_noise', 0.2, 0.0, 0.5),
                ('noise_clip', 0.5, 0.1, 1.0),
                ('policy_delay', 2, 1, 5),
                ('buffer_size', 1000000, 10000, 2000000)
            ]
        elif method == 'SAC':
            specific_params = [
                ('sac_lr', 0.0003, 0.00001, 0.01),
                ('tau', 0.005, 0.001, 0.01),
                ('alpha', 0.2, 0.01, 1.0),
                ('buffer_size', 1000000, 10000, 2000000)
            ]
        else:
            specific_params = []
        
        return common_params + specific_params
    
    def _update_method_params(self):
        """Update method parameters display"""
        # Clear existing
        for widget in self.params_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.method_params = {}
        
        # Get parameters
        params = self._get_method_params()
        
        # Compare mode - show all methods
        if self.compare_var.get():
            methods = ['PPO', 'TD3', 'SAC']
            for method_idx, method in enumerate(methods):
                # Method label
                method_label = tk.Label(self.params_scrollable_frame, 
                                      text=f"{method} Parameters",
                                      bg='#2b2b2b', fg=self.colors.get(method, 'white'),
                                      font=('Arial', 9, 'bold'))
                method_label.grid(row=method_idx * 20, column=0, columnspan=3,
                                sticky='w', padx=5, pady=(10, 5))
                
                # Get method params
                self.method_var.set(method)
                method_params = self._get_method_params()
                
                for i, (name, default, min_val, max_val) in enumerate(method_params):
                    row = method_idx * 20 + i + 1
                    
                    label = tk.Label(self.params_scrollable_frame,
                                   text=name.replace('_', ' ').title() + ':',
                                   bg='#2b2b2b', fg='white', font=('Arial', 9))
                    label.grid(row=row, column=0, sticky='w', padx=5, pady=2)
                    
                    var = tk.DoubleVar(value=default)
                    entry = tk.Entry(self.params_scrollable_frame, textvariable=var,
                                   width=10, bg='#404040', fg='white', font=('Arial', 9))
                    entry.grid(row=row, column=1, padx=5, pady=2)
                    
                    self.method_params[f"{method}_{name}"] = var
            
            # Restore original selection
            self.method_var.set('PPO')
        else:
            # Single method
            row_offset = 0
            method = self.method_var.get()
            
            # Add TD3 noise selection at top if TD3
            if method == 'TD3':
                noise_label = tk.Label(self.params_scrollable_frame,
                                     text="Noise Algorithms:",
                                     bg='#2b2b2b', fg='#4ECDC4', font=('Arial', 9, 'bold'))
                noise_label.grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=(5, 2))
                
                self.td3_noise_vars['exploration'] = tk.BooleanVar(value=True)
                self.td3_noise_vars['target_smoothing'] = tk.BooleanVar(value=False)
                
                expl_check = tk.Checkbutton(self.params_scrollable_frame,
                                          text="Exploration Noise",
                                          variable=self.td3_noise_vars['exploration'],
                                          bg='#2b2b2b', fg='white', selectcolor='#404040',
                                          font=('Arial', 9))
                expl_check.grid(row=1, column=0, columnspan=2, sticky='w', padx=20, pady=2)
                
                target_check = tk.Checkbutton(self.params_scrollable_frame,
                                            text="Target Policy Smoothing",
                                            variable=self.td3_noise_vars['target_smoothing'],
                                            bg='#2b2b2b', fg='white', selectcolor='#404040',
                                            font=('Arial', 9))
                target_check.grid(row=2, column=0, columnspan=2, sticky='w', padx=20, pady=2)
                
                row_offset = 3
            
            for i, (name, default, min_val, max_val) in enumerate(params):
                row = i + row_offset
                
                label = tk.Label(self.params_scrollable_frame,
                               text=name.replace('_', ' ').title() + ':',
                               bg='#2b2b2b', fg='white', font=('Arial', 9))
                label.grid(row=row, column=0, sticky='w', padx=5, pady=2)
                
                # Check if this is a TD3 scalable parameter
                if method == 'TD3' and name in ['actor_lr', 'critic_lr']:
                    # Create range mode checkbox
                    range_var = tk.BooleanVar(value=False)
                    self.td3_range_mode[name] = range_var
                    
                    range_check = tk.Checkbutton(self.params_scrollable_frame,
                                               text="Range",
                                               variable=range_var,
                                               bg='#2b2b2b', fg='#4ECDC4',
                                               selectcolor='#404040',
                                               font=('Arial', 8),
                                               command=lambda n=name: self._toggle_td3_range(n))
                    range_check.grid(row=row, column=2, padx=2, pady=2)
                    
                    # Create container for value/range inputs
                    input_frame = tk.Frame(self.params_scrollable_frame, bg='#2b2b2b')
                    input_frame.grid(row=row, column=1, padx=5, pady=2)
                    
                    # Single value (default)
                    var = tk.DoubleVar(value=default)
                    single_entry = tk.Entry(input_frame, textvariable=var,
                                          width=10, bg='#404040', fg='white', font=('Arial', 9))
                    single_entry.pack()
                    self.method_params[name] = var
                    
                    # Range inputs (hidden by default)
                    range_frame = tk.Frame(input_frame, bg='#2b2b2b')
                    
                    from_var = tk.DoubleVar(value=min_val)
                    to_var = tk.DoubleVar(value=max_val)
                    step_var = tk.DoubleVar(value=(max_val - min_val) / 5)
                    
                    tk.Label(range_frame, text="From:", bg='#2b2b2b', fg='white',
                           font=('Arial', 8)).grid(row=0, column=0, sticky='w')
                    tk.Entry(range_frame, textvariable=from_var, width=8,
                           bg='#404040', fg='white', font=('Arial', 8)).grid(row=0, column=1)
                    
                    tk.Label(range_frame, text="To:", bg='#2b2b2b', fg='white',
                           font=('Arial', 8)).grid(row=1, column=0, sticky='w')
                    tk.Entry(range_frame, textvariable=to_var, width=8,
                           bg='#404040', fg='white', font=('Arial', 8)).grid(row=1, column=1)
                    
                    tk.Label(range_frame, text="Step:", bg='#2b2b2b', fg='white',
                           font=('Arial', 8)).grid(row=2, column=0, sticky='w')
                    tk.Entry(range_frame, textvariable=step_var, width=8,
                           bg='#404040', fg='white', font=('Arial', 8)).grid(row=2, column=1)
                    
                    self.method_params[f"{name}_from"] = from_var
                    self.method_params[f"{name}_to"] = to_var
                    self.method_params[f"{name}_step"] = step_var
                    self.method_params[f"{name}_range_frame"] = (single_entry, range_frame)
                else:
                    var = tk.DoubleVar(value=default)
                    entry = tk.Entry(self.params_scrollable_frame, textvariable=var,
                                   width=10, bg='#404040', fg='white', font=('Arial', 9))
                    entry.grid(row=row, column=1, padx=5, pady=2)
                    
                    self.method_params[name] = var
        
        # Update scroll region
        self.params_scrollable_frame.update_idletasks()
        self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
    
    def _toggle_td3_range(self, param_name):
        """Toggle between single value and range mode for TD3 parameters"""
        if param_name in self.method_params and f"{param_name}_range_frame" in self.method_params:
            single_entry, range_frame = self.method_params[f"{param_name}_range_frame"]
            is_range = self.td3_range_mode[param_name].get()
            
            if is_range:
                single_entry.pack_forget()
                range_frame.pack()
            else:
                range_frame.pack_forget()
                single_entry.pack()
    
    def _on_method_change(self, event=None):
        """Handle method selection change"""
        if not self.compare_var.get():
            self._update_method_params()
    
    def _on_compare_change(self):
        """Handle compare mode change"""
        self._update_method_params()
    
    def _generate_td3_configs(self):
        """Generate all TD3 training configurations based on range and noise settings"""
        configs = []
        
        # Get noise types to test
        noise_types = []
        if self.td3_noise_vars.get('exploration', tk.BooleanVar(value=True)).get():
            noise_types.append('exploration')
        if self.td3_noise_vars.get('target_smoothing', tk.BooleanVar(value=False)).get():
            noise_types.append('target_smoothing')
        
        if not noise_types:
            noise_types = ['exploration']  # Default
        
        # Check if actor_lr or critic_lr are in range mode
        actor_lr_range = self.td3_range_mode.get('actor_lr', tk.BooleanVar(value=False)).get()
        critic_lr_range = self.td3_range_mode.get('critic_lr', tk.BooleanVar(value=False)).get()
        
        # Generate actor_lr values
        if actor_lr_range:
            actor_lr_from = self.method_params['actor_lr_from'].get()
            actor_lr_to = self.method_params['actor_lr_to'].get()
            actor_lr_step = self.method_params['actor_lr_step'].get()
            import numpy as np
            actor_lr_values = np.arange(actor_lr_from, actor_lr_to + actor_lr_step/2, actor_lr_step)
        else:
            actor_lr_values = [self.method_params['actor_lr'].get()]
        
        # Generate critic_lr values
        if critic_lr_range:
            critic_lr_from = self.method_params['critic_lr_from'].get()
            critic_lr_to = self.method_params['critic_lr_to'].get()
            critic_lr_step = self.method_params['critic_lr_step'].get()
            import numpy as np
            critic_lr_values = np.arange(critic_lr_from, critic_lr_to + critic_lr_step/2, critic_lr_step)
        else:
            critic_lr_values = [self.method_params['critic_lr'].get()]
        
        # Generate all combinations
        config_id = 0
        for noise_type in noise_types:
            for actor_lr in actor_lr_values:
                for critic_lr in critic_lr_values:
                    param_override = {
                        'actor_lr': actor_lr,
                        'critic_lr': critic_lr,
                        'config_id': config_id
                    }
                    configs.append(('TD3', param_override, noise_type))
                    config_id += 1
        
        return configs
    
    def _toggle_animation(self):
        """Toggle animation on/off"""
        self.animation_enabled = self.anim_var.get()
        if not self.animation_enabled:
            self.anim_canvas.delete("all")
        else:
            # Clear canvas and show message that animation will start with next training
            self.anim_canvas.delete("all")
            if not self.training:
                # Show info message on canvas
                self.anim_canvas.create_text(
                    180, 180, 
                    text="Animation will start\nwhen training begins",
                    fill='white', 
                    font=('Arial', 14),
                    justify='center'
                )
    
    def _update_animation_frame(self, frame):
        """Update animation canvas with new frame"""
        if not self.animation_enabled or frame is None:
            return
        
        try:
            # Convert to PIL Image (frame is already RGB from MuJoCo)
            image = Image.fromarray(frame.astype('uint8'))
            # Resize to fit canvas (360x360)
            image = image.resize((360, 360), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.anim_canvas.delete("all")
            self.anim_canvas.create_image(180, 180, image=photo)
            self.anim_canvas.image = photo
        except Exception as e:
            logger.warning(f"Animation update error: {e}")
    
    def _reset_environment(self):
        """Reset the environment"""
        self.env.reset()
        self.anim_canvas.delete("all")
        messagebox.showinfo("Reset", "Environment reset successfully!")
    
    def _create_policy(self, method_name, param_override=None, noise_type=None):
        """Create a policy instance"""
        state_dim = self.env.get_state_dim()
        action_dim = self.env.get_action_dim()
        
        # Get network params
        hidden_sizes = [
            int(self.network_params['hidden_layer_1'].get()),
            int(self.network_params['hidden_layer_2'].get())
        ]
        activation = self.activation_var.get()
        
        # Get method params
        if self.compare_var.get():
            prefix = f"{method_name}_"
            params = {k.replace(prefix, ''): v.get() 
                     for k, v in self.method_params.items() 
                     if k.startswith(prefix) and not k.endswith('_from') and not k.endswith('_to') and not k.endswith('_step') and not k.endswith('_range_frame')}
        else:
            params = {k: v.get() for k, v in self.method_params.items() 
                     if not k.endswith('_from') and not k.endswith('_to') and not k.endswith('_step') and not k.endswith('_range_frame')}
        
        # Apply parameter overrides (for TD3 loops)
        if param_override:
            params.update(param_override)
        
        # Create policy
        if method_name == 'PPO':
            lr = params.get('ppo_lr', 0.0003)
            policy = PPO(
                state_dim, action_dim, hidden_sizes,
                lr=lr,
                gamma=params.get('gamma', 0.99),
                clip_ratio=params.get('clip_ratio', 0.2),
                epochs=int(params.get('epochs', 10)),
                batch_size=int(params.get('batch_size', 256)),
                activation=activation
            )
        elif method_name == 'TD3':
            # Use provided noise_type or default to exploration
            td3_noise_type = noise_type if noise_type else 'exploration'
            
            policy = TD3(
                state_dim, action_dim, hidden_sizes,
                actor_lr=params.get('actor_lr', 0.0003),
                critic_lr=params.get('critic_lr', 0.0003),
                gamma=params.get('gamma', 0.99),
                tau=params.get('tau', 0.005),
                policy_noise=params.get('policy_noise', 0.2),
                noise_clip=params.get('noise_clip', 0.5),
                policy_delay=int(params.get('policy_delay', 2)),
                buffer_size=int(params.get('buffer_size', 1000000)),
                batch_size=int(params.get('batch_size', 256)),
                activation=activation,
                noise_type=td3_noise_type
            )
        elif method_name == 'SAC':
            policy = SAC(
                state_dim, action_dim, hidden_sizes,
                lr=params.get('sac_lr', 0.0003),
                gamma=params.get('gamma', 0.99),
                tau=params.get('tau', 0.005),
                alpha=params.get('alpha', 0.2),
                buffer_size=int(params.get('buffer_size', 1000000)),
                batch_size=int(params.get('batch_size', 256)),
                activation=activation
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        return policy
    
    def _train_method(self, method_name, param_override=None, noise_type=None, is_render_thread=False):
        """Train a single method"""
        # Create policy
        policy = self._create_policy(method_name, param_override, noise_type)
        
        # Create separate environment for this thread to avoid MuJoCo conflicts
        from walker_2d_logic import Walker2DEnvironment
        
        # Get environment params
        healthy_z_min, healthy_z_max = self.env_params['healthy_z_range']
        healthy_angle_min, healthy_angle_max = self.env_params['healthy_angle_range']
        
        # Enable rendering if this is the designated render thread
        render_mode = 'rgb_array' if (self.animation_enabled and is_render_thread) else None
        
        thread_env = Walker2DEnvironment(
            render_mode=render_mode,
            forward_reward_weight=self.env_params['forward_reward_weight'].get(),
            ctrl_cost_weight=self.env_params['ctrl_cost_weight'].get(),
            healthy_reward=self.env_params['healthy_reward'].get(),
            terminate_when_unhealthy=self.env_params['terminate_when_unhealthy'].get(),
            healthy_z_range=(healthy_z_min.get(), healthy_z_max.get()),
            healthy_angle_range=(healthy_angle_min.get(), healthy_angle_max.get()),
            reset_noise_scale=self.env_params['reset_noise_scale'].get()
        )
        
        # Create agent with thread-specific environment
        agent = Agent(thread_env, policy)
        
        # Create display name for this configuration
        display_name = method_name
        if method_name == 'TD3' and (param_override or noise_type):
            parts = [method_name]
            if noise_type:
                noise_short = 'Expl' if noise_type == 'exploration' else 'TPS'
                parts.append(noise_short)
            if param_override:
                if 'actor_lr' in param_override:
                    parts.append(f"a{param_override['actor_lr']:.5f}")
                if 'critic_lr' in param_override:
                    parts.append(f"c{param_override['critic_lr']:.5f}")
            display_name = '_'.join(parts)
        
        # Initialize data storage
        self.rewards_data[display_name] = []
        self.episode_counters[display_name] = 0
        
        # Training loop
        num_episodes = self.episodes_var.get()
        
        for episode in range(num_episodes):
            if self.stop_training:
                break
            
            # Run episode (rendering handled internally if needed)
            reward = agent.run_episode(train=True, render=False)
            
            # Store reward
            self.rewards_data[display_name].append(reward)
            self.episode_counters[display_name] = episode + 1
            
            # Update animation if enabled (only for designated render thread)
            if self.animation_enabled and is_render_thread and episode % 5 == 0:
                try:
                    frame = thread_env.render()
                    if frame is not None:
                        self.root.after(0, self._update_animation_frame, frame)
                except Exception as e:
                    logger.warning(f"Animation render failed: {e}")
            
            # Update plot periodically
            if episode % 10 == 0:
                self.root.after(0, self._update_plot)
        
        # Close thread-specific environment
        thread_env.close()
        logger.info(f"{display_name} training completed")
    
    def _start_training(self):
        """Start training process"""
        if self.training:
            messagebox.showwarning("Training", "Training already in progress!")
            return
        
        self.training = True
        self.stop_training = False
        self.rewards_data = {}
        self.episode_counters = {}
        self.config_colors = {}  # Reset color assignments
        
        # Clear plot
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_xlabel('Episode', color='white')
        self.ax.set_ylabel('Reward', color='white')
        self.ax.tick_params(colors='white')
        self.canvas.draw()
        
        # Create environment with current params
        from walker_2d_logic import Walker2DEnvironment
        
        render_mode = 'rgb_array' if self.animation_enabled else None
        
        # Get range params
        healthy_z_min, healthy_z_max = self.env_params['healthy_z_range']
        healthy_angle_min, healthy_angle_max = self.env_params['healthy_angle_range']
        
        self.env = Walker2DEnvironment(
            render_mode=render_mode,
            forward_reward_weight=self.env_params['forward_reward_weight'].get(),
            ctrl_cost_weight=self.env_params['ctrl_cost_weight'].get(),
            healthy_reward=self.env_params['healthy_reward'].get(),
            terminate_when_unhealthy=self.env_params['terminate_when_unhealthy'].get(),
            healthy_z_range=(healthy_z_min.get(), healthy_z_max.get()),
            healthy_angle_range=(healthy_angle_min.get(), healthy_angle_max.get()),
            reset_noise_scale=self.env_params['reset_noise_scale'].get()
        )
        
        # Start training threads
        if self.compare_var.get():
            methods = ['PPO', 'TD3', 'SAC']
            training_configs = [(m, None, None) for m in methods]  # (method, param_value, noise_type)
        else:
            method = self.method_var.get()
            
            # Handle TD3 parameter ranges and noise variations
            if method == 'TD3':
                training_configs = self._generate_td3_configs()
            else:
                training_configs = [(method, None, None)]
        
        self.training_threads = []
        for idx, config in enumerate(training_configs):
            method_name, param_override, noise_type = config
            # First thread handles rendering if animation is enabled
            is_render_thread = (idx == 0 and self.animation_enabled)
            thread = threading.Thread(target=self._train_method, 
                                    args=(method_name, param_override, noise_type, is_render_thread))
            thread.daemon = True
            thread.start()
            self.training_threads.append(thread)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_training)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def _monitor_training(self):
        """Monitor training threads"""
        for thread in self.training_threads:
            thread.join()
        
        self.training = False
        self.root.after(0, lambda: messagebox.showinfo("Training", "Training completed!"))
    
    def _stop_training_callback(self):
        """Stop training"""
        if not self.training:
            messagebox.showinfo("Training", "No training in progress!")
            return
        
        self.stop_training = True
        messagebox.showinfo("Training", "Stopping training...")
    
    def _update_plot(self):
        """Update the reward plot"""
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_xlabel('Episode', color='white')
        self.ax.set_ylabel('Reward', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('#2b2b2b')
        self.ax.spines['right'].set_color('#2b2b2b')
        
        # Plot each method
        z_order = 10
        for method_name, rewards in self.rewards_data.items():
            if len(rewards) == 0:
                continue
            
            episodes = np.arange(len(rewards))
            
            # Get unique color for this configuration
            if method_name not in self.config_colors:
                # Extract base method name
                base_method = method_name.split('_')[0]
                # If it's exactly a base method (PPO, TD3, SAC) and only one, use default color
                if method_name in self.colors and len([k for k in self.rewards_data.keys() if k.startswith(base_method)]) == 1:
                    color = self.colors[method_name]
                else:
                    # Assign next color from palette for all configs or when multiple configs exist
                    color_idx = len(self.config_colors) % len(self.color_palette)
                    color = self.color_palette[color_idx]
                self.config_colors[method_name] = color
            else:
                color = self.config_colors[method_name]
            
            # Plot raw rewards (light, background)
            self.ax.plot(episodes, rewards, alpha=0.3, color=color,
                        linewidth=0.5, zorder=z_order)
            
            # Plot moving average (bold, foreground)
            if len(rewards) >= 10:
                window = min(50, len(rewards) // 10)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                avg_episodes = episodes[window-1:]
                self.ax.plot(avg_episodes, moving_avg, color=color,
                           linewidth=2, label=method_name, zorder=z_order + 10)
            
            z_order -= 1
        
        # Legend
        if len(self.rewards_data) > 0:
            self.ax.legend(loc='lower left', facecolor='#2b2b2b',
                         edgecolor='white', labelcolor='white')
        
        self.canvas.draw()
    
    def _save_plot(self):
        """Save current plot to file"""
        if len(self.rewards_data) == 0:
            messagebox.showwarning("Save Plot", "No data to save!")
            return
        
        filename = f"walker2d_plot_{np.random.randint(10000)}.png"
        self.fig.savefig(filename, facecolor='#2b2b2b', edgecolor='white')
        messagebox.showinfo("Save Plot", f"Plot saved as {filename}")
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
