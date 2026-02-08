# Pusher RL Training Application

Reinforcement learning training application for the Pusher-v5 environment from Gymnasium (MuJoCo).

## Features

- **Multiple RL Algorithms**: DDPG, TD3, and SAC
- **Interactive GUI**: Tkinter interface with live grid
- **Real-time Visualization**: Live reward plotting with moving averages
- **Animation**: Optional environment rendering
- **Compare Mode**: Train and compare multiple algorithms simultaneously
- **Configurable Parameters**: Adjust environment and algorithm parameters on the fly

## Installation

### Prerequisites

- Python 3.8+
- MuJoCo (install following the official instructions)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
python pusher_app.py
```

### Environment Parameters

- **reward_near_weight** (default: 0.5)
- **reward_dist_weight** (default: 1.0)
- **reward_control_weight** (default: 0.1)

### Algorithm Parameters

All parameters are editable through the GUI.

#### Common Parameters
- **Episodes**: 200 (default)
- **Learning Rate**: 3e-4
- **Gamma**: 0.99
- **Hidden Dims**: 256,256 (comma-separated layer sizes)
- **Network Type**: mlp or cnn
- **Activation**: relu, tanh, leaky_relu

#### DDPG (Deep Deterministic Policy Gradient)
- **Tau**: 0.005
- **Buffer Size**: 100000
- **Batch Size**: 64
- **Exploration Noise**: 0.1
- **Actor Activation**: relu
- **Critic Activation**: relu

#### TD3 (Twin Delayed DDPG)
- **Tau**: 0.005
- **Policy Noise**: 0.2
- **Noise Clip**: 0.5
- **Policy Delay**: 2
- **Buffer Size**: 100000
- **Batch Size**: 64
- **Exploration Noise**: 0.1
- **Actor Activation**: relu
- **Critic Activation**: relu

#### SAC (Soft Actor-Critic)
- **Tau**: 0.005
- **Alpha**: 0.2
- **Buffer Size**: 100000
- **Batch Size**: 64
- **Train Frequency**: 4
- **Gradient Steps**: 1
- **Actor Activation**: relu
- **Critic Activation**: relu
- **Alpha From/To/Step**: range sweep for alpha
- **Buffer From/To/Step**: range sweep for replay buffer size
- **Gamma From/To/Step**: range sweep for gamma

## Files

- `pusher_logic.py`: Environment wrapper and RL algorithm implementations
- `pusher_gui.py`: Tkinter GUI interface
- `pusher_app.py`: Application entry point
- `requirements.txt`: Python dependencies
- `tests/test_pusher_logic.py`: Unit tests

## References

- [Pusher-v5 Documentation](https://gymnasium.farama.org/environments/mujoco/pusher/)
- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
