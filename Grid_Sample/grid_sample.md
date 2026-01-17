# Gridworld RL example with Tkinter GUI

## Goal: 
- implement a simple, self-contained reinforcement-learning Gridworld navigation demo.
- with both manual and agent-driven interaction via a Tkinter GUI.
- without any neural network!!

## Files to create
- `grid_logic.py`: content should be class for environment (`GridWorld`), classes for each algorithm (Monte Carlo V(s) `MonteCarloPolicies`, SARSA (`SarsaPolicies`), expected SARSA (`ExpSarsaPolicies`) and Q-learning `QlearningPolicies`), and class for `Agent` for injecting environment and algorithms to learn.
- `grid_gui.py`: Tkinter GUI to visualize the grid, run episodes step-by-step, train, and plot returns.
- `grid_app.py`: Entry point that imports GridWorld, Agent, MonteCarloPolicy, SarsaPolicy, ExpSarsaPolicy, QlearningPolicy from `grid_logic.py`, instatiate environment (gridworld), list of policies (MonteCarloPolicy, SarsaPolicy, ExpSarsaPolicy, QlearningPolicy), agent with injected environment and first policy as default (MonteCarloPolicy) and gui root (tk.Tk()). Finally instatiates the gui and inject root, environment, agent, policies and starts the gui.
- `README.md`: This documentation file
- `requirements.txt`: This requirements

## Grid World
- **Scalable Grid**: Default 5x3 grid, configurable to any size
- **Coordinate System**: Upper left corner (0,0), lower right corner (width-1, height-1)
- **Action System**: Numbered actions for RL compatibility
  - Action 0: Up (↑)
  - Action 1: Down (↓) 
  - Action 2: Left (←)
  - Action 3: Right (→)
- **Reward System**:
  - Reward: -1 per step, 0 at goal
- **Configurable Elements**:
  - Start position (default: 0,2)
  - End position (default: 4,2)
  - Blocked cells (default: (2,1) and (2,2))
  - Maximum steps per episode (default: 20)
- **Terminal/blocked behavior**: stepping into blocked cell—stay in place with reward -1?
- **Observation/state format**: tuple (x, y) consistent across env/policies/CSV.
- **Path Tracing**: Visual display of agent movement history
- **Step Limiting**: Episodes end when goal reached or max steps exceeded
- **Grid resizing**: validate start/goal/blocked within bounds; auto-correct!

## GUI: `grid_gui.py` (Tkinter)
- ### Layout:

  - **Control Panel Grid Conifguration**: top left
    - Grid Size (W x H)
    - Start Position (X, Y)
    - End Position (X, Y)
    - Blocked Cells ((X, Y), (X, Y) ...)

  - **Grid Display**: top right of `control panel grid configuration`
    - Visual representation of the grid world
    - Colored start and endpoint
    - Colored blocked cells in dark gray
    - Colored circle with 'A' for the current position of agent
    - Show grid position (X, Y) in light gray
    - On final episode draw directional arrows between consecutive states, so the path shows movement direction  

  - **Control Panel Episode Configuration**: left, below of `control panel grid configuration`
    - No of episodes to train, default 100
    - Max steps per episode, default 20 
    - Alpha (learning rate), default 0.1
    - Gamma (discount), default 0.9
    - Epsilon max, default 1.0
    - Epsilon min, default 0.05
    - Epsilon decay, default  
    - Selectable policy algorithm (monte carlo, sarsa, expected sarsa, q-learning)
    - Compare methods, single method: checkbox control, default compare methods
    - Button, background color yellow: Apply and reset (training and grid)

  - **Control Panel Action**: left, below of `control panel episode configuration`
    - Button: Run single step
    - Button: Run single epsiode
    - Button: Train and run (no of episodes)
    - episode progress bar n to max episode: shows running episodes progress
    - Button: Activate the dialog Value-Table 
    - Button: Activate the dialog Q-Table

  - **Dialog Value-Table**: 
    - Draws a copy of the gridworld design, and should show the latest V(s) in each cell. 
    - Button: Export trajectories of latest episode to csv, with state, action, next state, reward - include policy name in filename

  - **Dialog Q-Table**: 
    - Draws a copy of the gridworld design, and should show the latest max_a Q(s,a) in each cell.  
    - Button: Export trajectories of latest episode to csv, with state, action, next state, reward - include policy name in filename

  - **Live cumulative reward plot**: below grid, right of `control panel episode configuration`
    - Show the plot in current window
    - Show episodes as x and return as y
    - Show episodes return in compare mode: use light colors of red, green, blue and orange for each method.
    - Show moving averages in compare mode: use bold colors of red, green, blue, organge for each method.
    - Show a legend for methods: use the specified colors of method for legend keys.   
    - Button: to save plot image, include policy name in filename.
    - When "Compare methods" is requested, runs should be executed sequentially (one method after another).
    -  and plots should be side by side, comarable.

## Behaviour
 - ### Gui:
  - If new values applied, change params of environment, reset und change policy of agent from list of policies, if necessary. 

 - ### Methods 
  - Monte Carlo variant: every-visit; on-policy with epsilon greedy.

- #### epsilon decay ####
  - Monte-Carlo variant: epsilon = 1 / (1 + 0.001 * episode) with epsilon_min = 0.05
  - SARSA variant: epsilon = max(0.05, 1 / (1 + 0.001 * episode))
  - Expected SARSA variant: epsilon = max(0.05, 1 * exp(-0.0005 * episode))
  - Q-Learning variante: epsilon = max(0.05, 1 * exp(-0.005 * episode))

## Additions
- **Environment API**: `GridWorld(width:int,height:int,start:Tuple[int,int],goal:Tuple[int,int],blocked:List[Tuple[int,int]],max_steps:int,seed:Optional[int]=None)` with methods:
  - `reset() -> state`
  - `step(action:int) -> (next_state, reward:float, done:bool, info:dict)`

- **Agent / Policy API (recommended)**:
  - `Agent` should expose `start_episode()` / `end_episode()` hooks, `select_action(state) -> int`, and `update(state, action, reward, next_state, done)` so different algorithms can be plugged in.

- **CSV export format**: include columns `episode,step,state_x,state_y,action,next_state_x,next_state_y,reward,done`, include policy name in filename

- **Documentation**: include 'README.md' and document the grid_sample into it.

## Run

- Requires Python 3.9+.
- Requires for the reward plot: `matplotlib`.
- Run with: `python grid_app.py`

## Usage
- `grid_app.py` Entry point that imports GridWorld, Agent, MonteCarloPolicy, SarsaPolicy, ExpSarsaPolicy, QlearningPolicy from `grid_logic.py`, instatiate environment (gridworld), list of policies (MonteCarloPolicy, SarsaPolicy, ExpSarsaPolicy, QlearningPolicy), agent with injected environment and first policy as default (MonteCarloPolicy) and gui root (tk.Tk()). Finally instatiates the gui and inject root, environment, agent, policies and starts the gui.

## Tests
- unit tests for core algorithms and minimum coverage targets. 

# End.

