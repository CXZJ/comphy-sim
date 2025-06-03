# Bacterial Growth Simulation

A Python-based simulation of bacterial growth and nutrient diffusion in a 2D environment. This project models the complex interactions between bacterial populations and their nutrient environment using a combination of differential equations and Monte Carlo methods.

## Features

- **Interactive GUI**: Real-time visualization of bacterial movement and nutrient distribution
- **Monte Carlo Analysis**: Run multiple simulations to analyze statistical properties
- **Adjustable Parameters**: Fine-tune simulation parameters through an intuitive interface
- **Real-time Visualization**: Watch bacterial colonies grow and nutrients diffuse

## Key Parameters

- **Grid Size**: Size of the simulation grid (N×N)
- **Time Step (dt)**: Simulation time step (smaller values = more accurate but slower)
- **Diffusion (D)**: Nutrient diffusion coefficient
- **Growth Rate (r)**: Intrinsic bacterial growth rate
- **Carrying Capacity (K)**: Maximum sustainable bacterial population
- **Half-saturation (Ks)**: Nutrient concentration at half-maximal growth rate
- **Death Rate (γ)**: Probability of bacterial death per time step
- **Consumption Rate (β)**: Amount of nutrient consumed per bacterium
- **Step Size (σ)**: Standard deviation of bacterial random walk
- **Initial Conditions**: Set initial bacterial count and nutrient concentration

## Mathematical Model

The simulation combines several key equations:

1. **Nutrient Diffusion**:
   ```
   ∂C/∂t = D∇²C
   ```
   where C is nutrient concentration and D is diffusion coefficient

2. **Bacterial Growth**:
   ```
   P_grow = r * N * (1 - N/K) * (C/(C + Ks))
   ```
   combining logistic growth with Monod kinetics

3. **Nutrient Consumption**:
   ```
   C_new = max(C - β*dt, 0)
   ```

4. **Bacterial Movement**:
   Random walk with normal distribution:
   ```
   dy, dx ~ N(0, σ²)
   ```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Tkinter (usually comes with Python)

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd bacterial-growth-sim
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

## Usage

Run the simulation:
```bash
python main.py
```

### Controls

- **Start**: Begin the simulation
- **Pause**: Pause the simulation
- **Reset**: Reset the simulation with current parameters
- **Run MC**: Run Monte Carlo analysis with multiple simulations
- **Help**: View parameter explanations

### Monte Carlo Analysis

The Monte Carlo feature runs multiple independent simulations with the same parameters to analyze statistical properties of the system. Results show:
- Mean population over time
- Standard deviation of population
- Confidence intervals

## Contributing

Feel free to submit issues and enhancement requests!

## Physics Models Implemented

The simulation implements the following physical equations and models:

1. **Diffusion Equation**: ∂C/∂t = D∇²C
   - Models how nutrients diffuse through the medium
   - Implemented using Gaussian filtering

2. **Fick's First Law**: J = -D∇C
   - Describes the diffusive flux of nutrients
   - Gradient calculations drive nutrient flow

3. **Mean Squared Displacement**
   - Measures how far bacterial particles spread from the origin
   - MSD = <|r(t) - r(0)|²>

4. **Laplace's Equation**: ∇²C = 0
   - Used in steady-state nutrient distribution
   - Visualized through the Laplacian of the concentration field

5. **Monte Carlo Discrete Random-Walk**
   - Simulates diffusion-limited aggregation
   - Random walkers attach to the growing colony

6. **Box-Counting Fractal Dimension**
   - D = -lim_{ε→0} [log(N(ε))/log(ε)]
   - Quantifies the fractal nature of the bacterial colony

7. **Mass-Radius Scaling Relation**: M(r) ∝ r^D
   - Power-law relationship between colony mass and radius
   - The exponent D is the fractal dimension

8. **Harmonic-Measure Growth Probability**
   - Growth probability proportional to nutrient gradient magnitude
   - Models preferential growth at high-flux locations

## Features

- Interactive parameter selection
- Real-time visualization of bacterial growth
- Multiple analysis plots:
  - Colony size vs. time
  - Colony radius vs. time
  - Fractal dimension vs. time
  - Mean squared displacement vs. time
  - Mass-radius scaling (log-log plot)
  - Nutrient gradient visualization

- Controls for:
  - Pause/resume simulation
  - Adjust simulation speed
  - Reset simulation
  - Return to parameter selection

## Running the Simulation

1. Ensure you have the required packages installed:
   - numpy
   - matplotlib
   - scipy
   - tkinter (comes with Python)

2. Run the simulation:
   ```
   python revised_simulation.py
   ```

3. Enter simulation parameters or use the defaults

4. The simulation window will open, showing the growing bacterial colony

## How It Works

The simulation uses a grid-based approach where:

1. A central seed bacterium is placed at the start
2. Nutrients diffuse according to the diffusion equation
3. Bacteria grow through two mechanisms:
   - Deterministic growth based on nutrient concentration and gradients
   - Stochastic growth through random walkers (diffusion-limited aggregation)
4. Growth patterns emerge with fractal properties
5. Analysis tools calculate and display key metrics

The combination of these models produces complex, realistic bacterial growth patterns with fractal properties similar to those observed in nature.

## References

- Eden, M. (1961). *A two-dimensional growth process*. Berkeley Symp. on Math. Statist. and Prob.
- Witten, T. A., & Sander, L. M. (1981). *Diffusion-limited aggregation, a kinetic critical phenomenon*. Physical Review Letters.
- Matsushita, M., & Fujikawa, H. (1990). *Diffusion-limited growth in bacterial colony formation*. Physica A.
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. W. H. Freeman and Company. 
