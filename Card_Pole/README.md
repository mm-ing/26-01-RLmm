# CartPole RL Demo

A Tkinter-based reinforcement learning demo for the Gymnasium environment CartPole-v1 with live training animation and reward plots.

## Features
- Selectable methods: DQN, DDQN
- Compare methods with live reward plots
- Live Gym rendering during training
- Configurable: episodes, alpha, gamma, epsilon start/end/decay, step delay, max steps
- Neural network params: hidden layers, hidden units, batch size, replay buffer size, activation function
- Save plot image
- Cancel training safely (non-blocking GUI)

## Requirements
- Python 3.9+

## Install
From the repository root:
- Create a virtual environment if needed
- Install dependencies from Card_Pole/requirements.txt

## Run
From Card_Pole:
- python card_pole_app.py

## Notes
- Compare mode runs the selected methods in parallel threads.
- Only the selected method updates the live animation to keep the UI responsive.
