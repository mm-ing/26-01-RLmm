# Lunar Lander RL Demo

Reinforcement learning demo application for the LunarLander-v3 environment using various DQN-based methods.

## Features

- **Environment**: LunarLander-v3 (discrete and continuous variants)
- **Methods**: DQN, DDQN, Prioritized DDQN, Dueling DDQN, Distributional DQN, Noisy DQN, Rainbow
- **GUI**: Dark-themed Tkinter interface with live visualization
- **Training**: Multi-threaded training with live plotting
- **Comparison**: Compare multiple methods simultaneously

## Requirements

See `requirements.txt` for dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python lunar_lander_app.py
```

### Training Configuration

- **Episodes**: Number of training episodes
- **Alpha**: Learning rate
- **Gamma**: Discount factor
- **Epsilon**: Exploration parameters (start, end, decay)
- **Max steps**: Maximum steps per episode
- **Hidden layers/units**: Neural network architecture
- **Batch size**: Training batch size
- **Replay size**: Experience replay buffer size

### Environment Parameters

- **Continuous**: Use continuous action space (default: False)
- **Gravity**: Gravity strength (default: -10.0)
- **Enable wind**: Enable wind dynamics (default: False)
- **Wind power**: Wind strength (default: 15.0)
- **Turbulence power**: Turbulence strength (default: 1.5)

### Method-Specific Parameters

Each method has specific hyperparameters:

- **DQN/DDQN**: Target network update frequency
- **Prioritized DDQN**: Priority alpha, beta parameters
- **Distributional DQN**: Number of atoms, value range (Vmin, Vmax)
- **Noisy DQN**: Noise sigma
- **Rainbow**: Combination of all above parameters

## Controls

- **Train and run**: Start training with selected method(s)
- **Cancel learning**: Stop ongoing training
- **Reset**: Clear all results and reset UI
- **Save plot**: Export current plot as PNG
- **Disable/Enable animation**: Toggle environment rendering

## Methods

### DQN (Deep Q-Network)
Standard DQN with target network and experience replay.

### DDQN (Double DQN)
Uses Double Q-learning to reduce overestimation bias.

### Prioritized DDQN
Prioritized experience replay for more efficient learning.

### Dueling DDQN
Separate value and advantage streams in the network.

### Distributional DQN
Models the full distribution of returns.

### Noisy DQN
Noisy networks for exploration.

### Rainbow
Combines all improvements: DDQN, prioritized replay, dueling architecture, distributional learning, and noisy networks.

## Files

- `lunar_lander_logic.py`: Environment wrapper, policies, and agent
- `lunar_lander_gui.py`: GUI implementation
- `lunar_lander_app.py`: Entry point
- `tests/test_lunar_lander_logic.py`: Unit tests

## Tips for Learning

1. Start with DQN or DDQN for baseline performance
2. Increase hidden units (256-512) for better performance
3. Use larger replay buffers (100k-200k) for stable learning
4. Adjust target update frequency (100-200 steps)
5. Rainbow typically provides best performance but is slower

## License

MIT
