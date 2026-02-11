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
- Split GUI into 2 equal horizontal parts in relation 2 (upper) to 1 (lower)
#### Middle Section Upper (Method Control)
- Upper part contains a `Tab Control` with a tab for each requested algorithm.
- A tab should contain all editable parameters of the algorithm and its dependent neural networks.
#### Middle Section Lower (live Control Grid)
- Lower part contains a grid with selections and live parameters
##### Grid Columns (live Grid Control)
- !!! Method-Comparison: insert into the grid a new row for each selectable method !!!
- !!! Parameter-Comparison: if a parameter of a method contains a value-from (default value), value-to and step field and value-to and step is not empty, then create a new row in grid control with method name, parameter name and parameter values for value-from, each value between value-from and value-to dependent to step and value-to !!!
- ! Only one parameter can be tested at once!
###### Columns
- IsActive (rows are checkboxes)
- Selected method (rows are names)
- Current tested parameter (rows are names)
- Current tested value (rows are variant values)
- Episode/Total Episodes (rows are numeric/numeric)
- Step (rows are numeric)
- Reward (rows are numeric)
- Moving Average Reward (rows are numeric)
- Duration (rows are hh:mm:ss)
- Pause/Resume (rows are buttons)
- Animation (rows are radio buttons)
###### Rows
- IsActive (checkbox): if checked, method of this row has to be integrated into training process, otherwise skip it.
- Selected method name: show name from each selectable method
- Current tested parameter: show parameter if a parameter comparison is setted the name of this parameter has to be inserted here otherwise empty.
- Current tested value: show value if a parameter comparison is setted the current value of the selected parameter for comparison has to be inserted otherwise empty
- Episode/Total Episodes: show current running episode of total episodes
- Step: show current step of current episode 
- Reward: show current reward during training
- Moving Average Reward: show moving average during training
- Duration: show required accumulated time for current episodes of this method
- Pause/Resume: if pressed during training, method of this row stops working. Pressing again means working starts again from latest rewards.
- Animation: the animation can be active for only one animation. If checked the animation should be visible otherwise not visible

### Bottom Section (Plot)
#### Frame with buttons
- Create Button: Reset
- Create Button: Train
- Create Button: Cancel all learning (cancels all training)
- Create Button: Save plot (to image) to save the current plots to image into current folder (use method name plus value name if value exists)
- Reward plot across the entire width

#### Live Plot
- See GUI bottom section
- Use Matplotlib integration for live plotting
- Dark-themed plots
- Place legend allways lower left
- Select different colors for each method
- Show live reward plots for each selected method:
- with rewards per episode (light color, z-order: into background for every method) 
- with moving average (bold color into z-order: into foreground for every method, latest method most foreground)


### Animation (environment)
- Show environment (animation) see `GUI top section` and let it run live for the method for which visiblity is activated in grid column animation.
- Environment parameter configuration: all special params for selected animation should be visible and editable see `GUI top section`.
- !!!Let agent use the environment (animation) and its rewards to learn with the selected methods!!!
- Use OpenCV for environment rendering


### Learning Methods
- !!! All common parameters of selected methods should be visible and editable !!!
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
- All hyperparameters of neural networks for the different algorithms should be visible and editable


### Threading
- Run different methods in separate threads
- On compare mode run method threads parallel and show live on plot
- Non-blocking GUI during automatic pathfinding
- Show animation of current or latest selected method
- Safe thread communication with tkinter and gym


## Tests
- Create tests for each method
- fix program until agent get rewards and learns with methods