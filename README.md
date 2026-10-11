# Ant Colony Simulation

A high-performance, interactive Ant Colony Simulation built with PySide6, NumPy, and Numba. The application provides a real-time environment to observe and compare six distinct pathfinding and swarm-intelligence algorithms, supporting up to 2,000 agents simultaneously.

## Getting Started

**Prerequisites**
Python 3.8 or later is required.

**Installation**
Clone the repository and install the required dependencies:

```bash
pip install PySide6 numpy numba
```

**Execution**
Run the application using the following command:

```bash
python app.py
```

Note: The first launch will take a few seconds as Numba compiles the JIT-decorated functions to native machine code. Subsequent launches will use the cached compilation and start immediately.

---

## Interaction and Controls

### Input Bindings

**Mouse**
*   Left-click: Place a food source at the cursor position.
*   Right-click / Drag: Draw obstacle walls at the cursor position.

**Keyboard**
*   Space: Pause or resume the simulation.
*   R: Reset the simulation to its initial state.
*   P: Toggle pheromone rendering visibility.
*   C: Clear all placed obstacles.

### User Interface Parameters

The bottom panel provides real-time parameter adjustment via sliders. The mapping between the UI slider values and the internal simulation variables is as follows:

| Slider | Range | Internal Variable | Calculation | Precision |
| :--- | :--- | :--- | :--- | :--- |
| Speed | 1 - 10 | `sim_speed` | Direct value | Integer |
| Ants | 10 - 2000 | `n_active_ants` | Direct value | Integer |
| Ant Spd | 5 - 40 | `ant_speed_val` | value / 10.0 | 1 Decimal |
| Turn | 1 - 10 | `turn_rate_val` | value / 10.0 | 1 Decimal |
| Wander | 1 - 30 | `wander_noise_val` | value / 100.0 | 2 Decimals |
| Evap | 990 - 999 | `evap_rate_val` | value / 1000.0 | 3 Decimals |
| Diffuse | 1 - 15 | `diffuse_val` | value / 100.0 | 3 Decimals |
| MDP Gamma | 50 - 99 | `mdp_gamma` | value / 100.0 | 2 Decimals |

### Overlay Modes

The visualization overlay can be switched using the dropdown menu:

*   **Pheromones:** Renders the `food_pher` and `home_pher` grids using the selected color scheme (Classic, Thermal, Neon).
*   **MDP Values:** Renders the `V_food` and `V_nest` grids representing the computed optimal value functions.
*   **Stationary Dist:** Renders the `visit_grid` representing the Markov Chain stationary distribution.
*   **Q-Table:** Renders the maximum Q-value for each state cell from the reinforcement learning table.

---

## Algorithm Specifications

The simulation implements six distinct behavioral models for agent navigation. Agents switch behavior dynamically based on the selected algorithm while retaining their state (position, heading, food status).

### ACO (Standard)
Classic Ant Colony Optimization. Agents sample pheromone concentrations at three points (left sensor, center sensor, right sensor) separated by a sensor angle of 0.7 radians. The agent deterministically turns towards the sensor with the highest concentration. Pheromone deposition is active.

### Markov Chain
Agents sample pheromones identically to ACO, but select their direction stochastically. Transition probabilities are calculated as $P_L = S_L / (S_L + S_C + S_R + 10^{-6})$, $P_C = S_C / \text{total}$, and $P_R = S_R / \text{total}$. The agent draws a random number to choose its heading based on these probabilities. A dedicated visit grid tracks the stationary distribution of the swarm over time. Pheromone deposition is active.

### MDP Value Iteration
Agents do not deposit or follow pheromones. Instead, they follow the gradient of a pre-computed optimal Value Function. The simulation runs 20 iterations of Value Iteration per frame using a configurable discount factor ($\gamma$). Agents deterministically turn towards the adjacent grid cell with the highest expected value. Pheromone deposition is disabled.

### Q-Learning
Agents use reinforcement learning with a discrete action space (turn left, go straight, turn right) and a state space defined by the grid cell coordinates. The agent uses an $\epsilon$-greedy policy ($\epsilon = 0.15$) for exploration. Rewards are +100 for reaching food or the nest, -5 for hitting an obstacle, and -0.1 per step. The Q-table updates using a learning rate of 0.5 and a discount factor of 0.9. Pheromone deposition is disabled.

### Levy Flight
Agents perform random walks using a heavy-tailed step-length distribution. When the walk timer expires, the agent chooses a uniformly random heading and calculates a new jump length as $1 / U(0, 1)$, capped at 60 frames. While walking straight, agents apply a minor pheromone steering correction if the forward pheromone level is below 0.5. Pheromone deposition is modified to only occur during the straight-walking phase (timer > 10).

### Genetic Algorithm
Agents possess a 3-gene chromosome: [turn rate, sensor angle, wander noise]. An agent's fitness increases by 1 for picking up food and 2 for delivering it to the nest. If an agent's fitness exceeds 3, it reproduces by overwriting the genes of a random agent with fitness less than 1, applying a small uniform mutation (max 0.1 or 0.05 depending on the gene) to each gene. Pheromone deposition is active.

---

## Simulation Architecture

### Grid Configuration

The continuous 2D world is mapped onto a discrete grid to optimize spatial queries and pheromone diffusion.

*   **World Resolution:** 960 x 640 pixels
*   **Grid Cell Size:** 4 x 4 pixels
*   **Grid Dimensions:** 240 x 160 cells
*   **Nest Location:** (480, 320) pixels, Radius: 24 pixels
*   **Nest Pheromone:** Radius 60 pixels, Strength 80.0
*   **Food Capacity:** 500 units per source, Radius 12 pixels
*   **Obstacle Radius:** 14 pixels
*   **Agent Sensor Distance:** 20.0 pixels

### Pheromone Mechanics

Pheromone updates utilize a double-buffering technique to prevent directional bias. The `update_pheromones` function reads from the primary grids (`food_pher`, `home_pher`) and writes to temporary grids (`food_pher_tmp`, `home_pher_tmp`). The grid references are then swapped in Python.

Evaporation is applied first by multiplying the grid by the `evap_rate_val`. Diffusion is then calculated as a weighted average of the 8 neighboring cells, controlled by the `diffuse_val` coefficient. Values below 0.05 are zeroed out to maintain sparsity.

### Rendering Pipeline

Overlays are rendered by calculating RGBA NumPy arrays for the grid, creating a `QImage` from the buffer, and using `QPainter.drawImage` with `Qt.SmoothTransformation` to scale the 240x160 grid up to the 960x640 window. Agents are rendered natively via `QPainter` draw calls.

### Performance Optimization

Core simulation logic (agent stepping, pheromone evaporation/diffusion, MDP iteration) is compiled using Numba's `@njit(cache=True)` decorator. This allows the nested loops over thousands of agents and grid cells to execute at C-like speeds, enabling the simulation to maintain 60 FPS with 2,000 active agents.