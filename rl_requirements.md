# RL requirements

## Files to create
- use required RL methods
- replace `xxx` with the name of the application to create
- `xxx_logic.py`: content should be class for the selected environment, classes for each used algorithm and class for `Agent` for injecting environment and algorithms to learn.
- `xxx_gui.py`: Tkinter GUI to visualize the environment, run episodes step-by-step, train, and plot returns.
- `xxx_app.py`: Entry point that imports Environment, Agent and used RL-Methods from `grid_logic.py`, instatiate environment, list of policies, agent with injected environment and default policy and gui root (tk.Tk()). Finally instatiates the gui and inject root, environment, agent, policies and starts the gui.
- `README.md`: This documentation file
- `requirements.txt`: This requirements

## Main Features
- Animation support and optional rendering and environment parameter configuration in the animation section
- For each method a tab with configurable all algorithm specific parameters and all neural network related parameters.
- A live grid with selectable methods and selectable animation.
- A live plot showing reward plots.
- After starting training all activated methods in the "live grid" starts learning, updates animation smoothly (if animation is activated for this method) and shows reward plots for activated methods.
- If during training a new method was activated, this method added its rewards to the current running plot.
- If method parameter with value-from, value-to, step selection was activated it runs for all possible values at once and shows its current value within the grid.
- Comprehensive unit tests 

## GUI
### 3 Main Sections
- Split GUI into 3 identical horizontal parts
- Top part with animation
- Middle part with controls
- Buttom part with plots

### Top Section (Animation)
- Split animation section (top) into 2 vertical parts
- Left part show the visible animation
- Right part show animation editable specific settings

### Middle Section (Controls)
- Split GUI into 2 horizontal parts in relation 2 (upper) to 1 (lower)
#### Middle Section Upper
- Upper part contains tab control with tab for each requested algorithm.
- A Tab should contain all editable parameters of the algorithm and its dependent neural networks parameter configuration
#### Middle Section Lower (live Grid)
- Lower part contains a grid with selections and live parameters
##### Grid Columns
- IsActive (checkbox for selected method)
- Selected method name
- Current tested value (value of selected parameter for comparison, none if none selected)
- Episode/Total Episodes (current running episode of total episodes)
- Step (current step of current episode) 
- Moving Average Reward (from running method)
- Reward (current reward)
- Duration (required time for all episodes of this method)
- Animation (radio checkbox for witch method the animation should be visible, only on is selectable)

### Bottom Section (Plot)
#### Frame with buttons
- Create Button: Reset
- Create Button: Train
- Create Button: Cancel learning
- Create Button: Save plot (to image) to save the current plots to image into current folder (use method name plus value name if value exists)
- Reward plot across the entire width

#### Plot
- Use Matplotlib integration for live plotting
- Dark-themed plots
- Place legend allways lower left
- Select different colors for each method
- Show live reward plots for each selected method:
- with rewards per episode (light color, z-order: into background for every method) 
- with moving average (bold color into z-order: into foreground for every method, latest method most foreground)


## Algorithms
- Generate a tab for each requested algorithm (method)
- Search for all editable parameters of the algorithm and all editable parameters of the corresponding neural network and place them into the tab. 

### Scalable Method params:
- For the requested scalable method parameters replace value field with value-from, value-to and step, where value-from contains default value first.
- If the field value-to and step is filled, create a plot with method name and current value of this field and loop through the requested values.

### Learning Methods
- For user input use grid with 3 columns
- Make single method for compare mode selectable
- Possibility to pause single method in compare mode


## Animation


### Neural Networks
- !!!Use PyTorch-based neural networks (selectable MLP/CNN architecture)!!!
- Thread-safe training with proper locking
- Non-blocking GUI during training
- !!!Avoid loops, prefer numpy!!!
- Use numpy arrays if possible
- Neural network parameter configuration see GUI middle section upper part.
- Make Activation Methods (relu, tanh ...) selectable

### Learning
- !!! Make sure agents can reach every method!!!
- !!! Make sure that environment and agent work together!!!
- !!! Make sure that agent work together with the current method and makes learning progress!!!

## Threading
- Run different methods in separate threads
- On compare mode run method threads parallel and show live on plot
- Non-blocking GUI during automatic pathfinding
- Show animation of current or latest selected method
- Safe thread communication with tkinter and gym

## Tests
- Create and numerate tests for each method
- fix program until agent get rewards and learns with methods
