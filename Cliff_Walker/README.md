# CliffWalking RL Demo

A Tkinter-based reinforcement learning demo for the Gymnasium environment CliffWalking-v1 with live training animation and reward plots.

## Features
- Selectable methods: Q-learning, SARSA, Expected SARSA
- Compare methods with live reward plots
- Live Gym rendering during training
- Configurable: episodes, alpha, gamma, epsilon start/end, max steps, step delay
- Save plot image
- Cancel training safely (non-blocking GUI)

## Requirements
- Python 3.9+

## Install
From the repository root:
- Create a virtual environment if needed
- Install dependencies from Cliff_Walker/requirements.txt

## Run
From Cliff_Walker:
- python cliff_walker_app.py

## Notes
- Compare mode runs the selected methods in parallel threads.
- Only the first selected method updates the live animation to keep the UI responsive.
