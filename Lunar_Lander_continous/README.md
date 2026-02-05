# Lunar Lander Continuous RL Demo

Reinforcement learning demo application for the LunarLanderContinuous-v3 environment using various RL methods for continuous control.

## Features

- **Environment**: LunarLanderContinuous-v3 (continuous action space)
- **Methods**: Rainbow (DQN-based), A2C, TRPO, PPO
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
python lunar_lander_c_app.py
```

### Training Configuration

- **Episodes**: Number of training episodes
- **Alpha**: Learning rate
- **Gamma**: Discount factor
- **Max steps**: Maximum steps per episode
- **Hidden layers/units**: Neural network architecture
- **Batch size**: Training batch size
- **N steps**: Number of steps for n-step returns or trajectory length

### Environment Parameters

- **Continuous**: Use continuous action space (default: True)
- **Gravity**: Gravity strength (default: -10.0)
- **Enable wind**: Enable wind dynamics (default: False)
- **Wind power**: Wind strength (default: 15.0)
- **Turbulence power**: Turbulence strength (default: 1.5)

### Method-Specific Parameters

Each method has specific hyperparameters:

- **Rainbow**: Replay size, target update, atoms, Vmin/Vmax, noisy sigma
- **A2C**: Value coefficient, entropy coefficient, n-steps
- **TRPO**: Max KL divergence, damping, GAE lambda
- **PPO**: Clip epsilon, value coefficient, entropy coefficient, n-epochs, GAE lambda

## Controls

- **Train and run**: Start training with selected method(s)
- **Cancel learning**: Stop ongoing training
- **Reset**: Clear all results and reset UI
- **Save plot**: Export current plot as PNG
- **Disable/Enable animation**: Toggle environment rendering

## Methods

### Rainbow DQN
Combines multiple DQN improvements for continuous control (with discretized actions):
- Double Q-learning
- Prioritized experience replay
- Dueling architecture
- Distributional RL (C51)
- Noisy networks for exploration

### A2C (Advantage Actor-Critic)
Synchronous actor-critic method that uses the advantage function to reduce variance while maintaining policy gradient updates.

### TRPO (Trust Region Policy Optimization)
Guarantees monotonic policy improvement by constraining policy updates to a trust region defined by KL divergence.

### PPO (Proximal Policy Optimization)
Simplified version of TRPO that uses clipped surrogate objective instead of KL constraints, more stable and efficient.

## Files

- `lunar_lander_c_logic.py`: Environment wrapper, policies, and agent
- `lunar_lander_c_gui.py`: GUI implementation
- `lunar_lander_c_app.py`: Entry point
- `tests/test_lunar_lander_c_logic.py`: Unit tests

## Tips for Learning

1. **For Rainbow**: Requires more episodes due to DQN-based approach with discretized actions
2. **For A2C**: Good baseline, faster training but may be less stable
3. **For TRPO**: More stable but slower due to KL constraint computations
4. **For PPO**: Best balance of stability and efficiency, recommended for continuous control
5. Adjust learning rate (alpha) and n-steps based on method performance
6. Use compare mode to see which method works best for your configuration

## Notes

- Rainbow uses discretized actions for continuous control (experimental)
- Actor-critic methods (A2C, TRPO, PPO) directly output continuous actions
- PyTorch is used for all neural networks
- Matplotlib is used for live plotting
- Threading ensures non-blocking GUI during training
