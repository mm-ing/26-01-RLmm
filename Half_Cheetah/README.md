# HalfCheetah RL Training Application

Reinforcement Learning training application for the HalfCheetah-v5 environment from Gymnasium (Mujoco).

## Features

- **Multiple RL Algorithms**: PPO, TD3, and SAC
- **Interactive GUI**: Dark-themed Tkinter interface
- **Real-time Visualization**: Live reward plotting with moving averages
- **Animation**: Optional environment rendering
- **Compare Mode**: Train and compare multiple algorithms simultaneously
- **Configurable Parameters**: Adjust environment and algorithm parameters on the fly

## Installation

### Prerequisites

- Python 3.8+
- MuJoCo (install following [official instructions](https://github.com/openai/mujoco-py))

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
python half_cheetah_app.py
```

### Environment Parameters

- **Forward Reward Weight** (default: 1.0): Weight for forward velocity reward
- **Ctrl Cost Weight** (default: 0.1): Weight for control cost penalty
- **Reset Noise Scale** (default: 0.1): Scale of noise added on reset
- **Exclude Current Positions** (default: True): Whether to exclude x-position from observation

### Algorithm Parameters

All parameters are **fully editable** through the GUI.

#### Common Parameters
- **Episodes**: 5000 (default)
- **Learning Rate**: 3e-4
- **Gamma**: 0.99
- **Hidden Dims**: 256,256 (comma-separated layer sizes)
- **Activation**: relu (options: relu, tanh, leaky_relu)

#### PPO (Proximal Policy Optimization)
- **Clip Epsilon**: 0.2
- **Value Coefficient**: 0.5
- **Entropy Coefficient**: 0.01
- **Update Epochs**: 10
- **Network Activation**: relu

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

## Files

- `half_cheetah_logic.py`: Environment wrapper and RL algorithm implementations
- `half_cheetah_gui.py`: Tkinter GUI interface
- `half_cheetah_app.py`: Application entry point
- `requirements.txt`: Python dependencies
- `tests/test_half_cheetah_logic.py`: Unit tests

## Training Tips

1. **PPO**: Good general-purpose algorithm, stable and sample-efficient
2. **TD3**: Deterministic policy, good for fine control tasks
3. **SAC**: Maximum entropy framework, good exploration
4. Start with default parameters and adjust based on performance
5. Use compare mode to evaluate multiple algorithms

## References

- [HalfCheetah-v5 Documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
