# Bacterial Growth Simulation

This project simulates bacterial growth using fractal growth models and diffusion-limited aggregation (DLA). It implements several key physical equations to create a realistic model of how bacterial colonies form complex patterns.

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
