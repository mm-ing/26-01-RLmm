# Taxi RL Demo

A Tkinter-based reinforcement learning demo for the Gymnasium environment Taxi-v3 with live training animation and reward plots.

## Features
- Selectable methods: Q-learning, SARSA, Expected SARSA
- Compare methods with live reward plots
- Live Gym rendering during training
- Configurable: episodes, alpha, gamma, epsilon start/end, step delay, max steps
- Optional Is Raining toggle (adds a small step penalty)
- Save plot image
- Cancel training safely (non-blocking GUI)

## Requirements
- Python 3.9+

## Install
From the repository root:
- Create a virtual environment if needed
- Install dependencies from Taxi/requirements.txt

## Run
From Taxi:
- python taxi_app.py

## Notes
- Compare mode runs the selected methods in parallel threads.
- Only the selected method updates the live animation to keep the UI responsive.
