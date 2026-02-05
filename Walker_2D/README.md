# Walker2D RL Workbench

Reinforcement Learning application for training Walker2d-v5 environment using PyTorch-based algorithms.

## Overview

This application provides a complete RL training workbench with:
- **Walker2d-v5 environment** from Gymnasium MuJoCo
- **Three RL algorithms**: PPO, TD3, SAC
- **Dark-themed Tkinter GUI** with real-time visualization
- **Live reward plotting** with moving averages
- **Compare mode** for parallel algorithm training
- **Animation support** for environment rendering
- **Fully configurable parameters** for environment, algorithms, and neural networks

## Features

### Environment Parameters (Walker2D)
- `forward_reward_weight`: Weight for forward movement reward (default: 1.0)
- `ctrl_cost_weight`: Weight for control cost penalty (default: 0.001)
- `healthy_reward`: Reward for staying healthy (default: 1.0)
- `terminate_when_unhealthy`: Whether to terminate when unhealthy (default: True)
- `healthy_z_range`: Acceptable height range (default: 0.8 to 2.0)
- `healthy_angle_range`: Acceptable angle range (default: -1.0 to 1.0)
- `reset_noise_scale`: Scale of noise added on reset (default: 0.005)

### RL Algorithms

#### PPO (Proximal Policy Optimization)
- Clip ratio: 0.2
- Optimization epochs: 10
- Learning rate: 0.0003
- Gamma: 0.99
- Batch size: 256
- On-policy algorithm with actor-critic architecture
- Robust NaN detection and handling

#### TD3 (Twin Delayed DDPG)
- Actor learning rate: 0.0003
- Critic learning rate: 0.0003
- Tau (soft update): 0.005
- Policy noise: 0.2
- Noise clip: 0.5
- Policy delay: 2
- Buffer size: 1,000,000
- Batch size: 256
- **Advanced TD3 Features**:
  - **Parameter Range Mode**: Run loops with different learning rates
    - Actor learning rate: Configurable range (from, to, step)
    - Critic learning rate: Configurable range (from, to, step)
  - **Noise Algorithm Selection**: Choose one or both noise strategies
    - Exploration Noise: Adds noise during action selection
    - Target Policy Smoothing: Adds noise to target policy actions
  - **Automatic Loop Execution**: All combinations run in parallel threads
  - **Unique Plot Labels**: Each configuration labeled as TD3_[Noise]_a[lr]_c[lr]

#### SAC (Soft Actor-Critic)
- Learning rate: 0.0003
- Tau (soft update): 0.005
- Alpha (entropy): 0.2
- Buffer size: 1,000,000
- Batch size: 256
- Auto entropy tuning: Enabled
- Maximum entropy RL framework

### Neural Network Configuration
- **Architecture**: MLP (Multi-Layer Perceptron)
- **Hidden layers**: Configurable (default: [256, 256])
- **Activation functions**: relu, tanh, leaky_relu
- **Initialization**: Orthogonal initialization for stability
- All network parameters fully editable

### GUI Features
- **Animation**: Live environment rendering (enable/disable)
- **Method selection**: Choose PPO, TD3, or SAC
- **Compare mode**: Run multiple algorithms in parallel
- **Live plotting**: Reward curves with moving averages
- **Dark theme**: Professional dark color scheme
- **Scrollable parameters**: All 30+ parameters accessible
- **Control buttons**: Reset, Train, Cancel, Save Plot

### Compare Mode
- Train multiple algorithms simultaneously
- Different colors for each algorithm:
  - PPO: Red (#FF6B6B)
  - TD3: Teal (#4ECDC4)
  - SAC: Blue (#45B7D1)
- Individual parameter configuration per algorithm
- Real-time parallel visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training
1. Run the application:
```bash
python walker_2d_app.py
```

2. Configure environment parameters (optional)
3. Select an algorithm (PPO, TD3, or SAC)
4. Adjust algorithm parameters (optional)
5. Set number of episodes (default: 5000)
6. Click "Train" to start training

### TD3 Parameter Range Mode
1. Select "TD3" method
2. Check "Range" checkbox next to actor_lr or critic_lr (or both)
3. Set the range parameters:
   - **From**: Starting value (e.g., 0.0001)
   - **To**: Ending value (e.g., 0.0005)
   - **Step**: Increment (e.g., 0.0001)
4. Select noise algorithms:
   - Check "Exploration Noise" for exploration-based noise
   - Check "Target Policy Smoothing" for target smoothing noise
   - Select both to compare both strategies
5. Click "Train" - all combinations run automatically
6. Each configuration appears in the plot with a unique label

**Example**: If you set:
- actor_lr range: 0.0001 to 0.0003 (step 0.0001) = 3 values
- critic_lr range: 0.0002 to 0.0004 (step 0.0001) = 3 values  
- Both noise types selected
- **Result**: 3 × 3 × 2 = 18 training runs executed in parallel!

### Compare Mode
1. Enable "Compare Mode" checkbox
2. All three algorithms will be trained in parallel
3. Each algorithm has its own parameter configuration
4. Live plot shows all algorithms with different colors

### With Animation
1. Enable "Enable Animation" checkbox
2. Start training to see live Walker2D rendering
3. Animation updates every 10 episodes

### Save Results
- Click "Save Plot" to export the reward curve as PNG
- Saved as `walker2d_plot_<random>.png`

## Architecture

### Files
- `walker_2d_logic.py`: Environment wrapper, RL algorithms (PPO, TD3, SAC), Agent class
- `walker_2d_gui.py`: Tkinter GUI with animation, plotting, and controls
- `walker_2d_app.py`: Application entry point
- `tests/test_walker_2d_logic.py`: Comprehensive unit tests

### Class Structure
```
Walker2DEnvironment
├── Gymnasium Walker2d-v5 wrapper
├── Configurable parameters
└── Render support

PPO, TD3, SAC
├── PyTorch neural networks
├── Algorithm-specific logic
└── NaN protection (PPO)

Agent
├── Episode execution
├── Training coordination
└── Reward tracking

Walker2DGUI
├── Tkinter interface
├── Matplotlib integration
├── Threading for parallel training
└── Animation rendering
```

## Technical Details

### PPO Implementation
- Actor-Critic architecture
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Entropy bonus for exploration
- NaN detection and adaptive learning rate
- Orthogonal weight initialization

### TD3 Implementation
- Twin critic networks
- Delayed policy updates
- Target policy smoothing
- Exploration noise vs target smoothing noise (selectable)
- Soft target updates

### SAC Implementation
- Maximum entropy framework
- Twin critic networks
- Automatic entropy tuning
- Squashed Gaussian policy
- Continuous action support

### Threading and Stability
- **Non-blocking GUI** during training
- **Separate environment per thread** (prevents MuJoCo conflicts)
- **Thread-safe** reward data collection
- **Proper synchronization** with Tkinter
- **Action clipping** to valid range [-1, 1]
- **NaN/Inf detection** in actions (replaced with safe defaults)
- **Graceful error handling** in rendering and training
- **Environment cleanup** after training completion

### Animation Stability
- **Smooth rendering** with optimized frame updates
- **Frame resizing** (480x480 → 360x360) for efficient display
- **Reduced update frequency** (every 5 episodes) to prevent lag
- **Separate render environment** to avoid threading conflicts
- **Error recovery** when rendering fails
- **RGB format** directly from MuJoCo (no color conversion)

## Testing

Run unit tests:
```bash
python -m unittest tests.test_walker_2d_logic -v
```

Tests cover:
- Environment creation and configuration
- PPO training and action selection
- TD3 training and noise types
- SAC training and entropy tuning
- Agent episode execution
- All algorithm components

## Performance

- **Walker2D-v5**: 17 state dimensions, 6 action dimensions
- **PPO**: ~100-200 episodes for initial learning
- **TD3**: ~500-1000 episodes for stable performance
- **SAC**: ~300-600 episodes for good results
- **Training time**: Depends on hardware (GPU recommended)

## Troubleshooting

### MuJoCo Errors

**Error: `mj_makeConstraint: nefc under-allocation`**
- **Cause**: Multiple environments accessing MuJoCo simultaneously
- **Fix**: Each training thread now creates its own environment
- **Status**: ✅ Fixed automatically

**Warning: `Nan, Inf or huge value in QPOS`**
- **Cause**: Unstable actions causing simulation breakdown
- **Fix**: Actions are now clipped to [-1, 1] and NaN/Inf checked
- **Status**: ✅ Fixed automatically

### Animation Issues

**Animation not showing or flickering**
- Enable "Enable Animation" checkbox before training
- Animation updates every 5 episodes (reduces load)
- Only main training thread renders (prevents conflicts)
- Check that `opencv-python` is installed

**Animation lag or freezing**
- Reduce number of episodes
- Disable animation for faster training
- Close other GPU-intensive applications
- Animation is automatically downscaled to 360x360

### NaN Errors
- PPO includes automatic NaN detection
- Adaptive learning rate reduction on NaN
- Gradient clipping prevents explosions
- Action validation before environment step
- Check network parameter ranges

### Slow Training
- Reduce batch size
- Decrease buffer size
- Disable animation
- Use fewer episodes for testing
- Use GPU if available (PyTorch CUDA)

### Memory Issues
- Reduce buffer size (TD3, SAC)
- Decrease batch size
- Close other applications
- Use smaller hidden layer sizes
- Each thread has separate environment (memory efficient)

## License

MIT License

## References

- [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
