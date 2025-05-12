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
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bacterial Growth Simulation")
        self.create_parameter_ui()
        
    def create_parameter_ui(self):
        """Create the parameter input UI"""
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Default values
        self.grid_size = tk.StringVar(value="200")
        self.diffusion_coef = tk.StringVar(value="0.5")
        self.growth_rate = tk.StringVar(value="0.1")
        self.nutrient_replenish = tk.StringVar(value="0.01")
        self.max_time = tk.StringVar(value="100")
        self.time_step = tk.StringVar(value="0.2")
        self.dla_threshold = tk.StringVar(value="1.5")
        self.random_walk_steps = tk.StringVar(value="1000")
        
        # Parameter inputs
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
        
        row += 1
        ttk.Button(frame, text="Start Simulation", command=self.start_simulation).grid(row=row, column=0, columnspan=2, pady=20)
        
        # Add explanations for each parameter
        row += 1
        explanation_text = """
        Parameter Explanations:
        - Grid Size: Size of the simulation grid
        - Diffusion Coefficient: Controls how quickly nutrients diffuse
        - Growth Rate: Rate of bacterial colony expansion
        - Nutrient Replenishment: Rate at which nutrients are replenished
        - Maximum Time: Duration of the simulation
        - Time Step: Simulation time increment
        - DLA Threshold: Distance at which particles attach in DLA
        - Random Walk Steps: Maximum steps for each DLA particle
        """
        explanation = ttk.Label(frame, text=explanation_text, justify=tk.LEFT)
        explanation.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=10)
        
    def validate_params(self):
        """Validate input parameters"""
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
                
            return params
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return None
    
    def start_simulation(self):
        """Start the simulation with validated parameters"""
        params = self.validate_params()
        if params:
            self.root.withdraw()  # Hide parameter window
            visualization = SimulationVisualization(params)
            visualization.run()
            self.root.destroy()
    
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
        self.fig.colorbar(self.im_bacteria, ax=self.ax_bacteria, label='Density')
        
        self.im_nutrients = self.ax_nutrients.imshow(
            self.model.nutrients, cmap='hot', interpolation='nearest', 
            vmin=0, vmax=1, origin='lower'
        )
        self.ax_nutrients.set_title('Nutrient Concentration')
        self.fig.colorbar(self.im_nutrients, ax=self.ax_nutrients, label='Concentration')
        
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
        
        # Create empty gradient quiver plot
        self.ax_gradient.set_title('Nutrient Gradient')
        self.ax_gradient.set_xlabel('X')
        self.ax_gradient.set_ylabel('Y')
        
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
        self.anim = FuncAnimation(
            self.fig, self.update_frame, frames=None,
            interval=50, blit=False, repeat=False
        )
        
    def update_frame(self, frame):
        """Update animation frame"""
        if self.paused:
            return
            
        # Run multiple steps based on speed
        steps = max(1, int(self.run_speed))
        for _ in range(steps):
            if not self.model.update():
                self.paused = True
                break
                
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
        self.ax_fractal.set_ylim(1.0, 2.5)  # Reasonable range for 2D growth
        
        self.line_msd.set_data(self.model.times, self.model.mean_squared_displacements)
        self.ax_msd.relim()
        self.ax_msd.autoscale_view()
        
        # Update mass-radius scaling plot (log-log)
        self.ax_scaling.clear()
        self.ax_scaling.set_title('Mass-Radius Scaling')
        self.ax_scaling.set_xlabel('log(Radius)')
        self.ax_scaling.set_ylabel('log(Mass)')
        
        # Calculate current mass-radius scaling
        center = self.model.grid_size // 2
        max_radius = min(center, self.model.grid_size - center)
        radii = np.linspace(1, max_radius, 20)
        masses = []
        
        for r in radii:
            y, x = np.ogrid[-center:self.model.grid_size-center, -center:self.model.grid_size-center]
            mask = x*x + y*y <= r*r
            mass = np.sum(self.model.bacteria[mask])
            masses.append(mass)
            
        masses = np.array(masses)
        valid_points = masses > 0
        
        if np.sum(valid_points) > 1:
            log_radii = np.log(radii[valid_points])
            log_masses = np.log(masses[valid_points])
            self.ax_scaling.scatter(log_radii, log_masses, color='blue', s=30, alpha=0.7)
            
            # Fit line
            slope, intercept = np.polyfit(log_radii, log_masses, 1)
            fit_line = slope * log_radii + intercept
            self.ax_scaling.plot(log_radii, fit_line, 'r-', 
                                label=f'D = {slope:.2f}')
            self.ax_scaling.legend()
        
        # Update gradient plot
        self.ax_gradient.clear()
        self.ax_gradient.set_title('Nutrient Gradient')
        
        # Downsample for better visualization
        skip = max(1, self.model.grid_size // 40)
        X, Y = np.meshgrid(np.arange(0, self.model.grid_size, skip), 
                          np.arange(0, self.model.grid_size, skip))
        U = self.model.nutrient_gradient[::skip, ::skip, 0]
        V = self.model.nutrient_gradient[::skip, ::skip, 1]
        
        # Normalize vector lengths
        magnitude = np.sqrt(U**2 + V**2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1
        U = U / max_mag
        V = V / max_mag
        
        self.ax_gradient.quiver(X, Y, U, V, magnitude, cmap='viridis',
                               scale=30, pivot='mid')
        
        # Update statistics text
        self.stats_text.set_text(
            f'Time: {self.model.time:.1f}\n'
            f'Colony Size: {self.model.colony_sizes[-1]}\n'
            f'Radius: {self.model.radii[-1]:.2f}\n'
            f'Fractal Dim.: {self.model.fractal_dims[-1]:.3f}\n'
            f'Growth Rate: {self.model.growth_rate:.3f}\n'
            f'Diffusion Coef.: {self.model.diffusion_coef:.3f}\n\n'
            f'Physics Models:\n'
            f'- Diffusion Equation\n'
            f'- Fick\'s Law\n'
            f'- Laplace\'s Equation\n'
            f'- Random Walk\n'
            f'- Fractal Growth\n'
            f'- DLA'
        )
        
    def toggle_pause(self, event):
        """Toggle pause/resume simulation"""
        self.paused = not self.paused
        self.pause_button.label.set_text('Resume' if self.paused else 'Pause')
        
    def update_speed(self, val):
        """Update simulation speed"""
        self.run_speed = val
        
    def reset_simulation(self, event):
        """Reset the simulation"""
        self.model = BacterialGrowthModel(self.params)
        self.paused = False
        self.pause_button.label.set_text('Pause')
        
    def return_to_params(self, event):
        """Return to parameter input screen"""
        plt.close(self.fig)
        app = SimulationGUI()
        app.run()
        
    def run(self):
        """Run the visualization"""
        plt.show()
        
if __name__ == "__main__":
    app = SimulationGUI()
    app.run() 
