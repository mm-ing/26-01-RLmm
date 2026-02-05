# RL example with Tkinter GUI

## Files to create
- use required RL methods
- replace `xxx` with the name of the application to create
- `xxx_logic.py`: content should be class for the selected environment, classes for each used algorithm and class for `Agent` for injecting environment and algorithms to learn.
- `xxx_gui.py`: Tkinter GUI to visualize the environment, run episodes step-by-step, train, and plot returns.
- `xxx_app.py`: Entry point that imports Environment, Agent and used RL-Methods from `grid_logic.py`, instatiate environment, list of policies, agent with injected environment and default policy and gui root (tk.Tk()). Finally instatiates the gui and inject root, environment, agent, policies and starts the gui.
- `README.md`: This documentation file
- `requirements.txt`: This requirements

## Features:
- Compare mode for multiple algorithms
- Compare mode for single algorithm with different values of parameters.
- Run single algorithm in a loop with all values of given span value-from, value-to and step.
- Environment parameter configuration
- Neural network parameter configuration
- Animation support (optional rendering)
- Comprehensive unit tests 

### GUI
- Dark-themed Tkinter GUI
- For user input use grid with 3 columns
- Create Button: Reset
- Create Button: Train
- Create Button: Cancel learning
- Create Button: Save plot (to image) to save the current plots to image into current folder

### Animation (environment)
- Show small environment (animation) at `top of ui` and let it run live for the current selected method or during comare-mode for the latest method.
- All special params for selected animation should be visible and editable.
- !!!Let agent use the environment (animation) and its rewards to learn with the selected methods!!!
- OpenCV for environment rendering
- Create Button for enabling, disabling animation

### Learning Methods
- Make required methods selectable (dropdown) and add a compare check (checkbox) to show all methods parallel in live plot
- !!! All common parameters of selected methods should be visible and editable!!!
- !!! All special parameters of selected method should be visible and editable!!! 
- !!! In compare-mode all special parameters of each method should be visible and editable!!!
- For user input use grid with 3 columns
- Make single method for compare mode selectable
- Possibility to pause single method in compare mode

### Learning
- !!! Make sure agents can reach every method!!!
- !!! Make sure that environment and agent work together!!!
- !!! Make sure that agent work together with the current method and makes learning progress!!!

### Neural Networks
- !!!Use PyTorch-based neural networks (selectable MLP/CNN architecture)!!!
- Thread-safe training with proper locking
- Non-blocking GUI during training
- !!!Avoid loops, prefer numpy!!!
- Use numpy arrays if possible
- Make Activation Methods (relu, tanh ...) selectable
- All common parameters of neural networks of the different algorithms should be visible and editable
- Show special parameters of neural networks of selected method and make editable


### Live Plot
- Use Matplotlib integration for live plotting
- Height of plot should be useful to compare rewards
- Live reward plotting with moving averages
- Place plot at bottom of gui in full gui width
- Place legend allways lower left
- Dark-themed plots
- Show reward plots live to each developing method with reward per episode (light color, z-order: into background for every method) and moving average (bold color into z-order: into foreground for every method, latest method most foreground) with different colors for each method


### Threading
- Run different methods in separate threads
- On compare mode run method threads parallel and show live on plot
- Non-blocking GUI during automatic pathfinding
- Show animation of current or latest selected method
- Safe thread communication with tkinter and gym

## Tests
- Create tests for each method
- fix program until agent get rewards and learns with methods