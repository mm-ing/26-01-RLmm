# BipedalWalker Reinforcement Learning

Train and visualize reinforcement learning algorithms on the BipedalWalker-v3 environment from Gymnasium.

## Overview

This application provides an interactive GUI for training and comparing multiple RL algorithms on the BipedalWalker-v3 environment. The environment challenges an agent to control a bipedal walker to move forward while maintaining balance.

## Features

### Supported Algorithms
- **Rainbow DQN**: Deep Q-Network with rainbow improvements (adapted for continuous control via discretization)
- **A2C**: Advantage Actor-Critic
- **TRPO**: Trust Region Policy Optimization
- **PPO**: Proximal Policy Optimization
- **SAC**: Soft Actor-Critic

### GUI Features
- **Dark Mode Theme**: Easy on the eyes for long training sessions
- **Live Visualization**: Watch the environment in real-time during training
- **Live Animation**: Enable/disable live rendering of the BipedalWalker environment
- **Parameter Control**: Adjust all hyperparameters through the GUI
- **Compare Mode**: Train and compare multiple algorithms simultaneously
- **Live Plotting**: Real-time reward plots with moving averages
- **Save Results**: Export plots to images

### Environment Options
- **Standard Mode**: Regular BipedalWalker environment
- **Hardcore Mode**: More challenging terrain with obstacles

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python bipedal_walker_app.py
```

### Training Workflow

1. **Select Algorithm**: Choose from Rainbow, A2C, TRPO, PPO, or SAC
2. **Configure Parameters**: Adjust learning rate, gamma, network architecture, and algorithm-specific parameters
3. **Enable Animation** (optional): Check "Enable Animation" to see the walker in real-time during training/running
4. **Enable Compare Mode** (optional): Select multiple algorithms to train simultaneously
5. **Start Training**: Click "Train" to begin training
6. **Monitor Progress**: Watch the live reward plot, animation, and status updates
7. **Save Results**: Export the plot when training is complete

### Using Live Animation

The live animation feature displays the BipedalWalker environment in real-time:

1. **Enable Animation**: Check the "Enable Animation" checkbox in the left panel
2. **Run Episode**: Click "Run" to see a single episode with live rendering
3. **During Training**: The animation updates live while training is in progress
4. **Performance**: Animation may slow down training slightly due to rendering overhead
5. **Disable**: Uncheck "Enable Animation" for faster training without visualization

### GUI Layout

The GUI is organized into three columns:

**Left Panel:**
- Environment animation display
- Environment parameters (hardcore mode toggle)
- Algorithm selection
- Compare mode controls
- Training control buttons

**Middle Panel:**
- Common hyperparameters (episodes, learning rate, gamma, hidden dimensions)
- Algorithm-specific parameters
- Training status log

**Right Panel:**
- Live reward plot showing:
  - Individual episode rewards (light colored lines)
  - Moving average rewards (bold colored lines)
  - Multiple algorithms in compare mode

## Algorithm Parameters

### Common Parameters
- **Episodes**: Number of training episodes
- **Learning Rate**: Step size for optimizer
- **Gamma**: Discount factor for future rewards
- **Hidden Dims**: Neural network layer sizes (comma-separated)

### Rainbow DQN
- **Epsilon**: Initial exploration rate
- **Epsilon Min**: Minimum exploration rate
- **Epsilon Decay**: Rate of exploration decay
- **Buffer Size**: Replay buffer capacity
- **Batch Size**: Mini-batch size for training
- **Discrete Actions**: Number of discrete actions per dimension

### A2C
- **Value Coef**: Coefficient for value loss
- **Entropy Coef**: Coefficient for entropy bonus

### TRPO
- **Max KL**: Maximum KL divergence for policy update
- **Damping**: Damping coefficient for conjugate gradient
- **Value Coef**: Coefficient for value loss

### PPO
- **Clip Epsilon**: Clipping parameter for policy update
- **Value Coef**: Coefficient for value loss
- **Entropy Coef**: Coefficient for entropy bonus
- **Update Epochs**: Number of epochs per update

### SAC
- **Tau**: Soft update coefficient for target networks
- **Alpha**: Temperature parameter for entropy
- **Buffer Size**: Replay buffer capacity
- **Batch Size**: Mini-batch size for training
- **Train Freq**: Train every N steps (lower = more training, slower but potentially better learning)
- **Gradient Steps**: Number of gradient updates per training step (higher = more learning per update)

## Neural Networks

All algorithms use PyTorch for neural network implementation:
- **MLP**: Multi-Layer Perceptron for value/Q-networks
- **Actor-Critic**: Separate actor and critic networks for policy gradient methods
- **Customizable**: Hidden dimensions can be adjusted through the GUI

## Technical Details

### Threading
- Each algorithm runs in a separate thread when in compare mode
- Non-blocking GUI updates during training
- Thread-safe communication between training and visualization

### Plot Features
- Dark mode styling for better visibility
- Real-time updates every 5 episodes
- Moving average smoothing (window size: 20)
- Legend positioned in lower left
- Multiple colors for different algorithms
- Z-ordering: raw rewards in background, moving averages in foreground

## File Structure

```
Bipedal_Walker/
├── bipedal_walker_app.py       # Entry point
├── bipedal_walker_logic.py     # RL algorithms and environment
├── bipedal_walker_gui.py       # Tkinter GUI
├── bipedal_walker.md           # Project specification
├── README.md                   # This file
├── requirements.txt            # Dependencies
└── tests/
    └── test_bipedal_walker_logic.py  # Unit tests
```

## Environment Information

**State Space**: 24-dimensional continuous vector including:
- Hull angle and angular velocity
- Velocity components
- Joint angles and velocities
- Leg ground contact sensors
- Lidar measurements

**Action Space**: 4-dimensional continuous vector (each in [-1, 1]):
- Hip motor 1 (torque)
- Knee motor 1 (torque)
- Hip motor 2 (torque)
- Knee motor 2 (torque)

**Reward**: +300 points for moving forward, penalties for using motors, and -100 for falling

**Episode Termination**: 
- Walker falls (hull touches ground)
- Maximum 1600 steps reached

## Tips for Training

1. **Start with PPO or SAC**: These are generally the most stable algorithms for this environment
2. **Adjust Learning Rate**: Try values between 1e-4 and 1e-3 if training is unstable
3. **Use Compare Mode**: Train multiple algorithms to find the best one for your needs
4. **Hardcore Mode**: Only attempt after successfully training on standard mode
5. **Network Size**: Larger networks (e.g., 512, 512) may perform better but train slower
6. **SAC Performance Tuning**:
   - **Train Freq = 4** (default): Good balance of speed and learning
   - **Train Freq = 2**: Slower but more training updates
   - **Train Freq = 8**: Faster but fewer updates
   - **Gradient Steps = 1** (default): Standard
   - **Gradient Steps = 2-4**: More learning per update, slower but can improve sample efficiency

## Troubleshooting

- **Training not progressing**: Try reducing the learning rate or increasing network size
- **Unstable training**: Reduce learning rate or adjust value/entropy coefficients
- **GUI freezing**: Ensure animations are disabled during intensive training
- **Out of memory**: Reduce buffer size or batch size for Rainbow/SAC

## License

This project is for educational purposes.
