# Prompt: Build Pusher RL App

You are building a reinforcement learning GUI application for the MuJoCo Gym environment `Pusher-v5`. Use the requirements from `..\rl_workbench_init.md` as the base specification and apply the Pusher-specific details below.

## Core Goal
Create a full RL app for `Pusher-v5` that runs live with selectable methods and comparable methods, including a live reward plot and animation.

## Required Algorithms
Implement these methods as selectable algorithms:
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)

## Required Files
Create these files following the workbench spec:
- `pusher_logic.py`
- `pusher_gui.py`
- `pusher_app.py`
- `README.md`
- `requirements.txt`
- comprehensive tests

## Pusher Environment Parameters
Expose these parameters in the animation settings UI with defaults:
- `reward_near_weight` (default: 0.5)
- `reward_dist_weight` (default: 1.0)
- `reward_control_weight` (default: 0.1)

## Algorithm Parameter Controls
Expose all common and method-specific parameters (per workbench spec). Additionally, support these scalable parameters:
- SAC `alpha` with value-from / value-to / step
- SAC replay buffer size with value-from / value-to / step
- SAC `gamma` with value-from / value-to / step

## UI and Behavior
Follow the exact layout, threading, plotting, and live-grid behavior defined in `..\rl_workbench_init.md`. Ensure:
- Non-blocking GUI with safe thread communication.
- Live animation selectable per method.
- Live reward plot with per-method rewards and moving average.
- Compare mode runs methods in parallel and updates plots.
- Agents can reach every method and learn using the environment.

## Technical Constraints
- Use PyTorch-based MLP/CNN networks with selectable activations.
- Prefer numpy over explicit loops where possible.
- Thread-safe training and GUI updates.
- OpenCV rendering for the environment animation.

## Tests
Provide unit tests for each method and verify learning progress (agent receives rewards and improves).
