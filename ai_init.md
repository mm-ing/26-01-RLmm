# RL example with Tkinter GUI

## Files to create
- use required RL methods
- replace `xxx` with the name of the application to create
- `xxx_logic.py`: content should be class for environment, classes for each used algorithm and class for `Agent` for injecting environment and algorithms to learn.
- `xxx_gui.py`: Tkinter GUI to visualize the environment, run episodes step-by-step, train, and plot returns.
- `xxx_app.py`: Entry point that imports Environment, Agent and used RL-Methods from `grid_logic.py`, instatiate environment, list of policies, agent with injected environment and default policy and gui root (tk.Tk()). Finally instatiates the gui and inject root, environment, agent, policies and starts the gui.
- `README.md`: This documentation file
- `requirements.txt`: This requirements

## Features
- Show gym animation at top of ui and let it run live to learning
- Make required methods selectable and add a compare check to show live reward plots of all selectable methods
- Show editable episodes (default 1000), alpha (default 0.2), gamma (default 0.8), epsilon-start (default 1.0), epsilon-end (default 0.05), step-delay (default 0.0)
- Create Button: Train and run
- Create Button: Cancel learning
- Show live reward plots with reward per episode (light color) and moving average (bold color) with different colors for each method
- Create Button: Save plot (to image) to save the current plots to image

### Threading
- Different methods run in separate threads
- Non-blocking GUI during automatic pathfinding
- Safe thread communication with tkinter and gym