# RL example with Tkinter GUI

## Files to create
- use required RL methods
- replace `xxx` with the name of the application to create
- `xxx_logic.py`: content should be class for the selected environment, classes for each used algorithm and class for `Agent` for injecting environment and algorithms to learn.
- `xxx_gui.py`: Tkinter GUI to visualize the environment, run episodes step-by-step, train, and plot returns.
- `xxx_app.py`: Entry point that imports Environment, Agent and used RL-Methods from `grid_logic.py`, instatiate environment, list of policies, agent with injected environment and default policy and gui root (tk.Tk()). Finally instatiates the gui and inject root, environment, agent, policies and starts the gui.
- `README.md`: This documentation file
- `requirements.txt`: This requirements

## Inital Features

### GUI
- Use darkmode theme for tkinter ui
- Use grid with 3 columns for user inputs

### Animation (environment)
- Show gym animation at top of ui and let it run live for the current selected method or during comare-mode for the latest method.
- All special params for selected animation should be visible and editable.
- Create Button for enabling, diabling animation
- !!!Let agent use the environment (animation) and its rewards to learn with the selected methods!!!

### Learning Methods
- Make required methods selectable (dropdown) and add a compare check (checkbox) to show all methods parallel in live plot
- All common parameters of selected methods should be visible and editable
- All special parameters of selected method should be visible and editable, 
- In compare-mode all special parameters of each method should be visible and editable
- Make methods for compare mode selectable
- !!!Avoid loops, prefer numpy!!!
- Let be replay buffer numpy arrays if available
- !!!assure agent reach every method and can learn from method!!!

### Neural Networks
- !!! Use Pytorch for neural networks !!!

- All common parameters of neural networks of selected method should be visible and editable
- All special parameters of neural networks of selected method should be visible and editable
- In compare-mode all special parameters of neural networks of each method should be visible and editable
- Create Button: Reset, train and run
- Create Button: Cancel learning

### Live Plot
- !!! Use matplotlib for plots !!!
- place legend allways upper left
- Use darkmode for plots
- Show reward plots live to each developing method with reward per episode (light color, z-order: into background for every method) and moving average (bold color into z-order: into foreground for every method, latest method most foreground) with different colors for each method
- Create Button: Save plot (to image) to save the current plots to image into current folder

## Threading
- Run different methods in separate threads
- On compare mode run method threads parallel and show live on plot
- Non-blocking GUI during automatic pathfinding
- Show animation of current or latest selected method
- Safe thread communication with tkinter and gym

## Tests
- create tests for each method
- fix each method until test pass