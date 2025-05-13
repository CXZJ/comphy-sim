import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from scipy.ndimage import gaussian_filter, laplace
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from scipy.spatial.distance import cdist

class BacterialGrowthModel:
    """Core physics model for bacterial growth simulation using fractal growth and DLA"""
    
    def __init__(self, params):
        # Basic parameters
        self.grid_size = params['grid_size']
        self.diffusion_coef = params['diffusion_coef']
        self.growth_rate = params['growth_rate']
        self.nutrient_replenish = params['nutrient_replenish']
        self.max_time = params['max_time']
        self.time_step = params['time_step']
        self.dla_threshold = params['dla_threshold']
        self.random_walk_steps = params['random_walk_steps']
        
        # Initialize grids
        self.bacteria = np.zeros((self.grid_size, self.grid_size))
        self.nutrients = np.ones((self.grid_size, self.grid_size))
        self.nutrient_gradient = np.zeros((self.grid_size, self.grid_size, 2))
        
        # Place initial seed
        center = self.grid_size // 2
        self.bacteria[center, center] = 1
        
        # Analysis data
        self.time = 0
        self.times = [0]
        self.colony_sizes = [1]
        self.radii = [0]
        self.fractal_dims = [1.0]
        self.mean_squared_displacements = [0]
        self.boundary_cells = [(center, center)]
        
    def compute_nutrient_diffusion(self):
        """Diffusion Equation (∂C/∂t = D∇²C)
        Using Gaussian filter as an approximation to the diffusion equation"""
        self.nutrients = gaussian_filter(self.nutrients, sigma=self.diffusion_coef)
        
        # Compute Laplacian (∇²C) for visualization
        self.nutrient_laplacian = laplace(self.nutrients)
        
    def compute_nutrient_gradient(self):
        """Fick's First Law: J = -D∇C
        Computing the gradient of nutrient concentration"""
        grad_y, grad_x = np.gradient(self.nutrients)
        self.nutrient_gradient = np.stack((grad_x, grad_y), axis=-1)
        
        # Compute diffusive flux magnitude (|J| = D|∇C|)
        self.flux_magnitude = self.diffusion_coef * np.sqrt(grad_x**2 + grad_y**2)
        
    def update_boundary_cells(self):
        """Find all boundary cells (empty cells adjacent to bacteria)"""
        self.boundary_cells = []
        bacteria_indices = np.where(self.bacteria > 0)
        
        for i in range(len(bacteria_indices[0])):
            x, y = bacteria_indices[0][i], bacteria_indices[1][i]
            
            # Check all 8 neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                        self.bacteria[nx, ny] == 0):
                        self.boundary_cells.append((nx, ny))
        
        # Remove duplicates
        self.boundary_cells = list(set(self.boundary_cells))
        
    def compute_growth_probabilities(self):
        """Harmonic-Measure Growth Probability
        P(x) ∝ |∇C(x)| at boundary sites"""
        probs = []
        for x, y in self.boundary_cells:
            # Growth probability depends on:
            # 1. Local nutrient concentration
            # 2. Magnitude of nutrient gradient (approximating harmonic measure)
            nutrient_factor = self.nutrients[x, y]
            gradient_factor = self.flux_magnitude[x, y]
            
            # Combined growth probability
            prob = self.growth_rate * nutrient_factor * (1 + gradient_factor)
            probs.append(prob)
            
        return probs
        
    def random_walk_particle(self):
        """Monte Carlo Discrete Random-Walk
        Simulation of DLA via particle random walk"""
        # Initialize random walker at the edge of the domain
        theta = 2 * np.pi * np.random.random()
        radius = self.grid_size * 0.45  # Start slightly inside the boundary
        center = self.grid_size // 2
        
        x = int(center + radius * np.cos(theta))
        y = int(center + radius * np.sin(theta))
        
        # Keep within bounds
        x = max(0, min(self.grid_size-1, x))
        y = max(0, min(self.grid_size-1, y))
        
        # Random walk until hitting a bacterial cell or exceeding steps
        for _ in range(self.random_walk_steps):
            # Calculate distances to bacterial cells
            bacteria_coords = np.argwhere(self.bacteria > 0)
            if len(bacteria_coords) == 0:
                return None
                
            # If adjacent to bacterial cell, attach
            min_distance = np.min(cdist(np.array([[x, y]]), bacteria_coords))
            if min_distance <= self.dla_threshold:
                return (x, y)
                
            # Random step (von Neumann neighborhood)
            step = np.random.choice(4)
            if step == 0: x += 1
            elif step == 1: x -= 1
            elif step == 2: y += 1
            else: y -= 1
            
            # Periodic boundary
            x = x % self.grid_size
            y = y % self.grid_size
            
        return None  # Failed to attach within step limit
        
    def calculate_mean_squared_displacement(self):
        """Mean Squared Displacement: MSD = <|r(t) - r(0)|²>
        Measures how far particles have moved from the origin"""
        bacteria_coords = np.argwhere(self.bacteria > 0)
        if len(bacteria_coords) == 0:
            return 0
            
        center = np.array([self.grid_size//2, self.grid_size//2])
        squared_distances = np.sum((bacteria_coords - center)**2, axis=1)
        return np.mean(squared_distances)
        
    def calculate_radius(self):
        """Maximum radius of the bacterial colony from center"""
        bacteria_coords = np.argwhere(self.bacteria > 0)
        if len(bacteria_coords) == 0:
            return 0
            
        center = np.array([self.grid_size//2, self.grid_size//2])
        distances = np.sqrt(np.sum((bacteria_coords - center)**2, axis=1))
        return np.max(distances)
        
    def calculate_mass_radius_scaling(self):
        """Mass-Radius Scaling Relation: M(r) ∝ r^D
        Where D is the fractal dimension"""
        center = self.grid_size // 2
        max_radius = min(center, self.grid_size - center)
        
        # Calculate mass (number of bacteria) within different radii
        radii = np.linspace(1, max_radius, 20)
        masses = []
        
        for r in radii:
            # Create a circular mask
            y, x = np.ogrid[-center:self.grid_size-center, -center:self.grid_size-center]
            mask = x*x + y*y <= r*r
            # Count bacteria within radius
            mass = np.sum(self.bacteria[mask])
            masses.append(mass)
            
        # Convert to logs for power-law fitting
        log_radii = np.log(radii[masses > 0])
        log_masses = np.log(np.array(masses)[masses > 0])
        
        if len(log_radii) > 1:
            # Fit power law: M(r) ∝ r^D, so log(M) = D*log(r) + const
            slope, _ = np.polyfit(log_radii, log_masses, 1)
            return slope  # This slope is the fractal dimension D
        else:
            return 1.0  # Default for early growth
    
    def calculate_box_dimension(self):
        """Box-Counting Fractal Dimension
        D = -lim_{ε→0} [log(N(ε))/log(ε)]
        where N(ε) is the number of boxes of size ε needed to cover the set"""
        def box_count(box_size):
            count = 0
            for i in range(0, self.grid_size, box_size):
                for j in range(0, self.grid_size, box_size):
                    if np.any(self.bacteria[i:i+box_size, j:j+box_size]):
                        count += 1
            return count
        
        # Use box sizes that are powers of 2
        box_sizes = [2**i for i in range(1, int(np.log2(self.grid_size//4))+1)]
        counts = [box_count(size) for size in box_sizes]
        
        # Fit power law
        valid_indices = np.where(np.array(counts) > 0)[0]
        if len(valid_indices) > 1:
            log_sizes = np.log(np.array(box_sizes)[valid_indices])
            log_counts = np.log(np.array(counts)[valid_indices])
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            return -slope  # Negative because N(ε) ∝ ε^(-D)
        else:
            return 1.0
        
    def update(self):
        """Update the simulation state for one time step"""
        if self.time >= self.max_time:
            return False
            
        # 1. Diffusion of nutrients
        self.compute_nutrient_diffusion()
        self.compute_nutrient_gradient()
        
        # 2. Identify potential growth sites
        self.update_boundary_cells()
        
        # 3. Growth via two mechanisms
        
        # a) Concentration and gradient-based growth (deterministic + stochastic)
        growth_probs = self.compute_growth_probabilities()
        for (x, y), prob in zip(self.boundary_cells, growth_probs):
            if np.random.random() < prob * self.time_step:
                self.bacteria[x, y] = 1
                self.nutrients[x, y] *= 0.5  # Consume nutrients
        
        # b) DLA particle attachment (purely stochastic)
        for _ in range(max(1, int(self.growth_rate * 10 * self.time_step))):
            attachment_site = self.random_walk_particle()
            if attachment_site:
                x, y = attachment_site
                self.bacteria[x, y] = 1
                self.nutrients[x, y] *= 0.5  # Consume nutrients
        
        # 4. Nutrient replenishment
        self.nutrients += self.nutrient_replenish * self.time_step
        self.nutrients = np.clip(self.nutrients, 0, 1)  # Keep in valid range
        
        # 5. Update analysis metrics
        self.time += self.time_step
        self.times.append(self.time)
        
        colony_size = np.sum(self.bacteria)
        self.colony_sizes.append(colony_size)
        
        radius = self.calculate_radius()
        self.radii.append(radius)
        
        fractal_dim = self.calculate_box_dimension()
        self.fractal_dims.append(fractal_dim)
        
        msd = self.calculate_mean_squared_displacement()
        self.mean_squared_displacements.append(msd)
        
        return True
        
class SimulationGUI:
    """GUI for the bacterial growth simulation"""
    
    def __init__(self, initial_params=None):
        self.root = tk.Tk()
        self.root.title("Bacterial Growth Simulation Parameters")
        # Store initial params if provided
        self.initial_params = initial_params 
        self.create_parameter_ui()
        
    def create_parameter_ui(self):
        """Create the parameter input UI"""
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Default values
        defaults = {
            'grid_size': "200", 'diffusion_coef': "0.5", 'growth_rate': "0.1",
            'nutrient_replenish': "0.01", 'max_time': "100", 'time_step': "0.2",
            'dla_threshold': "1.5", 'random_walk_steps': "1000",
            'num_trials': "10" # Added default for MC trials
        }
        
        # Use initial_params if available, otherwise use defaults
        current_values = defaults
        if self.initial_params: 
            # Ensure all expected keys are present, falling back to default if missing
            current_values = {key: str(self.initial_params.get(key, defaults[key])) 
                              for key in defaults}

        # Create StringVars using the determined current values
        self.grid_size = tk.StringVar(value=current_values['grid_size'])
        self.diffusion_coef = tk.StringVar(value=current_values['diffusion_coef'])
        self.growth_rate = tk.StringVar(value=current_values['growth_rate'])
        self.nutrient_replenish = tk.StringVar(value=current_values['nutrient_replenish'])
        self.max_time = tk.StringVar(value=current_values['max_time'])
        self.time_step = tk.StringVar(value=current_values['time_step'])
        self.dla_threshold = tk.StringVar(value=current_values['dla_threshold'])
        self.random_walk_steps = tk.StringVar(value=current_values['random_walk_steps'])
        self.num_trials = tk.StringVar(value=current_values['num_trials']) # Added StringVar for MC trials

        
        # Parameter inputs layout
        row = 0
        ttk.Label(frame, text="Grid Size (50-500):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.grid_size).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        ttk.Label(frame, text="Diffusion Coefficient (0.1-2.0):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.diffusion_coef).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        ttk.Label(frame, text="Growth Rate (0.01-1.0):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.growth_rate).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        ttk.Label(frame, text="Nutrient Replenishment (0.001-0.1):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.nutrient_replenish).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        ttk.Label(frame, text="Maximum Time (10-1000):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.max_time).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        ttk.Label(frame, text="Time Step (0.01-1.0):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.time_step).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        ttk.Label(frame, text="DLA Threshold (1.0-3.0):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.dla_threshold).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        ttk.Label(frame, text="Random Walk Steps (100-5000):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.random_walk_steps).grid(row=row, column=1, padx=5, pady=5)
        
        # Add Monte Carlo Trials Input
        row += 1
        ttk.Label(frame, text="MC Trials (>=2):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.num_trials).grid(row=row, column=1, padx=5, pady=5)

        # Buttons Frame for better layout
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=row + 1, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Start Simulation", command=self.start_simulation).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Run MC Analysis", command=self.run_mc_analysis).pack(side=tk.LEFT, padx=10) # New Button
        
        # Add explanations for each parameter
        row += 2
        explanation_text = """
        Parameter Explanations:
        - Grid Size: Size of the simulation grid (pixels)
        - Diffusion Coefficient: Rate of nutrient spread (sigma for Gaussian filter)
        - Growth Rate: Base probability factor for cell division
        - Nutrient Replenishment: Rate nutrients are added back per time step
        - Maximum Time: Total simulation duration (arbitrary units)
        - Time Step: Simulation time increment per update cycle
        - DLA Threshold: Max distance for random walker attachment (pixels)
        - Random Walk Steps: Max steps before a DLA walker gives up
        - MC Trials: Number of simulation runs for statistical error analysis
        """
        explanation = ttk.Label(frame, text=explanation_text, justify=tk.LEFT)
        explanation.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=10)
        
    def validate_params(self, validate_trials=False): # Added flag
        """Validate input parameters. Optionally validate num_trials."""
        try:
            params = {
                'grid_size': int(self.grid_size.get()),
                'diffusion_coef': float(self.diffusion_coef.get()),
                'growth_rate': float(self.growth_rate.get()),
                'nutrient_replenish': float(self.nutrient_replenish.get()),
                'max_time': float(self.max_time.get()),
                'time_step': float(self.time_step.get()),
                'dla_threshold': float(self.dla_threshold.get()),
                'random_walk_steps': int(self.random_walk_steps.get())
            }
            if validate_trials:
                params['num_trials'] = int(self.num_trials.get())
            
            # Validate ranges
            if not (50 <= params['grid_size'] <= 500):
                raise ValueError("Grid size must be between 50 and 500")
            if not (0.1 <= params['diffusion_coef'] <= 2.0):
                raise ValueError("Diffusion coefficient must be between 0.1 and 2.0")
            if not (0.01 <= params['growth_rate'] <= 1.0):
                raise ValueError("Growth rate must be between 0.01 and 1.0")
            if not (0.001 <= params['nutrient_replenish'] <= 0.1):
                raise ValueError("Nutrient replenishment must be between 0.001 and 0.1")
            if not (10 <= params['max_time'] <= 1000):
                raise ValueError("Maximum time must be between 10 and 1000")
            if not (0.01 <= params['time_step'] <= 1.0):
                raise ValueError("Time step must be between 0.01 and 1.0")
            if not (1.0 <= params['dla_threshold'] <= 3.0):
                raise ValueError("DLA threshold must be between 1.0 and 3.0")
            if not (100 <= params['random_walk_steps'] <= 5000):
                raise ValueError("Random walk steps must be between 100 and 5000")

            if validate_trials:
                if not (params['num_trials'] >= 2):
                     raise ValueError("Number of MC Trials must be at least 2 for statistics")
                
            return params
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return None
    
    def start_simulation(self):
        """Start the simulation visualization""" # Clarified purpose
        # Validate regular params, not num_trials
        params = self.validate_params(validate_trials=False) 
        if params:
            # Add num_trials to params dict just for consistency when returning
            try: 
                params['num_trials'] = int(self.num_trials.get())
            except ValueError:
                params['num_trials'] = 10 # Default if invalid
                
            self.root.withdraw()  # Hide parameter window
            visualization = SimulationVisualization(params)
            visualization.run()
            # If visualization window is closed, exit the GUI
            self.root.destroy()

    def run_mc_analysis(self): # New method for MC Analysis
        """Run multiple simulation trials and calculate statistics."""
        # Validate all params, including num_trials
        params = self.validate_params(validate_trials=True)
        if not params:
            return # Validation failed

        num_trials = params['num_trials']
        results = {
            'radii': [],
            'fractal_dims': [],
            'colony_sizes': [],
            'msds': []
        }

        # --- Show progress/info message --- 
        progress_window = tk.Toplevel(self.root)
        progress_window.title("MC Analysis Running")
        progress_label = ttk.Label(progress_window, text=f"Running {num_trials} trials... Please wait.")
        progress_label.pack(padx=20, pady=20)
        self.root.update_idletasks() # Ensure window appears
        
        try:
            # --- Run Trials --- 
            for i in range(num_trials):
                print(f"Running Trial {i+1}/{num_trials}...") # Console progress
                model = BacterialGrowthModel(params)
                # Run simulation loop without visualization
                while model.update():
                    pass 
                # Collect final results
                results['radii'].append(model.radii[-1])
                results['fractal_dims'].append(model.fractal_dims[-1])
                results['colony_sizes'].append(model.colony_sizes[-1])
                results['msds'].append(model.mean_squared_displacements[-1])
            
            # --- Calculate Statistics --- 
            stats_text = f"Monte Carlo Analysis Results ({num_trials} Trials):\n\n"
            for key, values in results.items():
                if values:
                    mean_val = np.mean(values)
                    std_dev = np.std(values)
                    # Standard Error of the Mean = std_dev / sqrt(N)
                    sem = std_dev / np.sqrt(num_trials) 
                    stats_text += f"Final {key.replace('_', ' ').title()}: {mean_val:.3f} ± {sem:.3f} (SEM)\n"
                else:
                    stats_text += f"Final {key.replace('_', ' ').title()}: No data\n"
            
            progress_window.destroy() # Close progress window
            messagebox.showinfo("MC Analysis Complete", stats_text)
            
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("MC Analysis Error", f"An error occurred during analysis: {e}")
            print(f"Error during MC Analysis: {e}") # Also print to console

    def run(self):
        """Run the GUI main loop"""
        self.root.mainloop()
        
class SimulationVisualization:
    """Visualization of the bacterial growth simulation"""
    
    def __init__(self, params):
        self.params = params
        self.model = BacterialGrowthModel(params)
        self.paused = False
        self.current_time = 0
        self.run_speed = 1.0  # Animation speed multiplier
        
        self.setup_figure()
        self.setup_animation()
        
    def setup_figure(self):
        """Set up the matplotlib figure and subplots"""
        plt.rcParams.update({'font.size': 10})
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(3, 4, width_ratios=[2, 2, 1, 1], 
                                   height_ratios=[1, 1, 1],
                                   wspace=0.3, hspace=0.3)
        
        # Main visualization plots
        self.ax_bacteria = self.fig.add_subplot(gs[0:2, 0])
        self.ax_nutrients = self.fig.add_subplot(gs[0:2, 1])
        
        # Analysis plots
        self.ax_size = self.fig.add_subplot(gs[0, 2])
        self.ax_radius = self.fig.add_subplot(gs[0, 3])
        self.ax_fractal = self.fig.add_subplot(gs[1, 2])
        self.ax_msd = self.fig.add_subplot(gs[1, 3])
        self.ax_scaling = self.fig.add_subplot(gs[2, 0])
        self.ax_gradient = self.fig.add_subplot(gs[2, 1])
        
        # Status and statistics text area
        self.ax_stats = self.fig.add_subplot(gs[2, 2:])
        self.ax_stats.axis('off')
        
        # Initialize main plots
        self.im_bacteria = self.ax_bacteria.imshow(
            self.model.bacteria, cmap='viridis', interpolation='nearest',
            vmin=0, vmax=1, origin='lower'
        )
        self.ax_bacteria.set_title('Bacterial Colony')
        self.cbar_bacteria = self.fig.colorbar(self.im_bacteria, ax=self.ax_bacteria, label='Density')
        
        self.im_nutrients = self.ax_nutrients.imshow(
            self.model.nutrients, cmap='hot', interpolation='nearest', 
            vmin=0, vmax=1, origin='lower'
        )
        self.ax_nutrients.set_title('Nutrient Concentration')
        self.cbar_nutrients = self.fig.colorbar(self.im_nutrients, ax=self.ax_nutrients, label='Concentration')
        
        # Initialize analysis plots
        self.line_size, = self.ax_size.plot([], [], 'b-')
        self.ax_size.set_title('Colony Size')
        self.ax_size.set_xlabel('Time')
        self.ax_size.set_ylabel('Cell Count')
        
        self.line_radius, = self.ax_radius.plot([], [], 'r-')
        self.ax_radius.set_title('Colony Radius')
        self.ax_radius.set_xlabel('Time')
        self.ax_radius.set_ylabel('Radius')
        
        self.line_fractal, = self.ax_fractal.plot([], [], 'g-')
        self.ax_fractal.set_title('Fractal Dimension')
        self.ax_fractal.set_xlabel('Time')
        self.ax_fractal.set_ylabel('Dimension')
        
        self.line_msd, = self.ax_msd.plot([], [], 'm-')
        self.ax_msd.set_title('Mean Squared Displacement')
        self.ax_msd.set_xlabel('Time')
        self.ax_msd.set_ylabel('MSD')
        
        # Initially empty scaling plot
        self.ax_scaling.set_title('Mass-Radius Scaling')
        self.ax_scaling.set_xlabel('log(Radius)')
        self.ax_scaling.set_ylabel('log(Mass)')
        
        # --- Initialize Nutrient Gradient Quiver --- 
        self.ax_gradient.set_title('Nutrient Gradient (Vector Field)')
        self.ax_gradient.set_xlabel('X')
        self.ax_gradient.set_ylabel('Y')
        
        # Calculate initial downsampled grid and gradient
        skip = max(1, self.params['grid_size'] // 40)
        self.X_grid, self.Y_grid = np.meshgrid(np.arange(0, self.params['grid_size'], skip), 
                                               np.arange(0, self.params['grid_size'], skip))
        # Compute initial gradient for initialization
        self.model.compute_nutrient_gradient() # Ensure gradient is calculated once initially
        U_init = self.model.nutrient_gradient[::skip, ::skip, 0]
        V_init = self.model.nutrient_gradient[::skip, ::skip, 1]
        Mag_init = np.sqrt(U_init**2 + V_init**2)

        # Create the quiver plot with initial data and correct number of points
        self.quiver_gradient = self.ax_gradient.quiver(
            self.X_grid.ravel(), self.Y_grid.ravel(), # Positions (correct size now)
            U_init.ravel(), V_init.ravel(),           # Initial vectors
            Mag_init.ravel(),                         # Initial magnitudes for color
            cmap='coolwarm', scale=30, pivot='mid'
        )
        self.ax_gradient.set_xlim(0, self.params['grid_size'])
        self.ax_gradient.set_ylim(0, self.params['grid_size'])
        self.cbar_gradient = self.fig.colorbar(self.quiver_gradient, ax=self.ax_gradient, label='Gradient Magnitude')
        # Set initial color limits
        if np.any(Mag_init > 0):
             self.cbar_gradient.mappable.set_clim(vmin=np.min(Mag_init), vmax=np.max(Mag_init))
        else:
             self.cbar_gradient.mappable.set_clim(vmin=0, vmax=1) # Default range

        # Statistics text
        self.stats_text = self.ax_stats.text(
            0.05, 0.95, '', transform=self.ax_stats.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Add control buttons
        self.button_ax = plt.axes([0.85, 0.03, 0.1, 0.04])
        self.pause_button = Button(self.button_ax, 'Pause')
        self.pause_button.on_clicked(self.toggle_pause)
        
        self.speed_ax = plt.axes([0.70, 0.03, 0.1, 0.04])
        self.speed_slider = Slider(self.speed_ax, 'Speed', 0.1, 5.0, valinit=1.0)
        self.speed_slider.on_changed(self.update_speed)
        
        self.reset_ax = plt.axes([0.55, 0.03, 0.1, 0.04])
        self.reset_button = Button(self.reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_simulation)
        
        self.param_ax = plt.axes([0.40, 0.03, 0.1, 0.04])
        self.param_button = Button(self.param_ax, 'Parameters')
        self.param_button.on_clicked(self.return_to_params)
        
    def setup_animation(self):
        """Set up the animation"""
        # Calculate the number of frames based on max_time and time_step
        num_frames = int(self.params['max_time'] / self.params['time_step'])
        
        self.anim = FuncAnimation(
            self.fig, self.update_frame, 
            frames=num_frames, # Provide explicit frames
            interval=50, blit=False, repeat=False,
            cache_frame_data=False # Explicitly disable caching to avoid potential issues
        )
        
    def update_frame(self, frame):
        """Update animation frame"""
        if self.paused:
             # Even if paused, return artists to prevent potential freezing with blit=True later
            return (self.im_bacteria, self.im_nutrients, self.line_size, self.line_radius, 
                    self.line_fractal, self.line_msd, self.quiver_gradient)
            
        # Run multiple simulation steps based on speed
        steps_to_run = max(1, int(self.run_speed))
        simulation_running = True
        for _ in range(steps_to_run):
            if not self.model.update():
                self.paused = True # Stop simulation updates
                self.pause_button.label.set_text('Resume') # Update button
                simulation_running = False
                break # Exit inner loop
        
        if not simulation_running and self.model.time >= self.params['max_time']:
            # If simulation ended naturally, ensure we don't process invalid frame numbers
             return (self.im_bacteria, self.im_nutrients, self.line_size, self.line_radius, 
                    self.line_fractal, self.line_msd, self.quiver_gradient)

        # --- Update Plots --- 
        # Update main visualization
        self.im_bacteria.set_array(self.model.bacteria)
        self.im_nutrients.set_array(self.model.nutrients)
        
        # Update analysis plots
        self.line_size.set_data(self.model.times, self.model.colony_sizes)
        self.ax_size.relim()
        self.ax_size.autoscale_view()
        
        self.line_radius.set_data(self.model.times, self.model.radii)
        self.ax_radius.relim()
        self.ax_radius.autoscale_view()
        
        self.line_fractal.set_data(self.model.times, self.model.fractal_dims)
        self.ax_fractal.relim()
        self.ax_fractal.autoscale_view()
        self.ax_fractal.set_ylim(1.0, 2.5) # Reasonable range for 2D growth
        
        self.line_msd.set_data(self.model.times, self.model.mean_squared_displacements)
        self.ax_msd.relim()
        self.ax_msd.autoscale_view()
        
        # Update mass-radius scaling plot (log-log)
        self.ax_scaling.clear()
        self.ax_scaling.set_title('Mass-Radius Scaling (log-log)')
        self.ax_scaling.set_xlabel('log(Radius)')
        self.ax_scaling.set_ylabel('log(Mass)')
        center = self.params['grid_size'] // 2
        max_radius = min(center, self.params['grid_size'] - center)
        radii = np.linspace(1, max_radius, 20)
        masses = []
        for r in radii:
            y, x = np.ogrid[-center:self.params['grid_size']-center, -center:self.params['grid_size']-center]
            mask = x*x + y*y <= r*r
            mass = np.sum(self.model.bacteria[mask])
            masses.append(mass)
        masses = np.array(masses)
        valid_points = masses > 0
        if np.sum(valid_points) > 1:
            log_radii = np.log(radii[valid_points])
            log_masses = np.log(masses[valid_points])
            self.ax_scaling.scatter(log_radii, log_masses, color='blue', s=30, alpha=0.7, label='Data (log(Mass) vs log(Radius))')
            slope, intercept = np.polyfit(log_radii, log_masses, 1)
            fit_line = slope * log_radii + intercept
            self.ax_scaling.plot(log_radii, fit_line, 'r-', label=f'Fit: slope = D ≈ {slope:.2f}')
            self.ax_scaling.legend()
            self.ax_scaling.grid(True)
        else:
             self.ax_scaling.text(0.5, 0.5, 'Not enough data for scaling analysis', horizontalalignment='center', verticalalignment='center', transform=self.ax_scaling.transAxes)
        
        # --- Update gradient plot --- 
        skip = max(1, self.params['grid_size'] // 40)
        U = self.model.nutrient_gradient[::skip, ::skip, 0]
        V = self.model.nutrient_gradient[::skip, ::skip, 1]
        magnitude = np.sqrt(U**2 + V**2)
        
        self.quiver_gradient.set_UVC(U.ravel(), V.ravel(), magnitude.ravel())
        
        if np.any(magnitude > 0):
             self.quiver_gradient.autoscale()
             self.cbar_gradient.mappable.set_clim(vmin=np.min(magnitude), vmax=np.max(magnitude))
        else:
             self.cbar_gradient.mappable.set_clim(vmin=0, vmax=1)

        # Update statistics text
        self.stats_text.set_text(
            f'Time: {self.model.time:.1f} / {self.params["max_time"]}\n' # Show max time
            f'Colony Size: {self.model.colony_sizes[-1]}\n'
            f'Radius: {self.model.radii[-1]:.2f}\n'
            f'Fractal Dim.: {self.model.fractal_dims[-1]:.3f}\n'
            f'Growth Rate: {self.model.growth_rate:.3f}\n'
            f'Diffusion Coef.: {self.model.diffusion_coef:.3f}\n\n'
            f'Physics Models:\n'
            f'- Diffusion Equation\n'
            f'- Fick\'s Law (J = -D∇C)\n'
            f'- Laplace\'s Equation (∇²C)\n'
            f'- Random Walk (DLA)\n'
            f'- Fractal Growth (Box Count & Scaling)\n'
            f'- Harmonic Measure Growth'
        )
        
        # Return list of updated artists
        return (self.im_bacteria, self.im_nutrients, self.line_size, self.line_radius, 
                self.line_fractal, self.line_msd, self.quiver_gradient, self.stats_text)
    
    def toggle_pause(self, event):
        """Toggle pause/resume simulation"""
        self.paused = not self.paused
        self.pause_button.label.set_text('Resume' if self.paused else 'Pause')
    
    def update_speed(self, val):
        """Update simulation speed"""
        self.run_speed = val
        
    def reset_simulation(self, event):
        """Reset the simulation using the current parameters"""
        self.model = BacterialGrowthModel(self.params) # Re-initialize model
        self.paused = False
        self.pause_button.label.set_text('Pause')
        # Don't explicitly call update_frame or draw_idle here
        # FuncAnimation should handle redrawing the initial state on the next cycle
        # Or force a single update if FuncAnimation isn't running/restarting smoothly
        try:
            # Try to seek to the beginning if animation is active
            self.anim.frame_seq = self.anim.new_frame_seq()
        except AttributeError:
             # If FuncAnimation hasn't started fully or has issues, manually update once
             self.update_frame(0)
             self.fig.canvas.draw_idle()
             
    def return_to_params(self, event):
        """Return to parameter input screen, passing current parameters"""
        plt.close(self.fig) # Close the plot window
        # Pass the parameters used for the *current* visualization run
        app = SimulationGUI(initial_params=self.params) 
        app.run()
    
    def run(self):
        """Run the visualization"""
        plt.show()

if __name__ == "__main__":
    # Start with no initial parameters the first time
    app = SimulationGUI(initial_params=None) 
    app.run()
