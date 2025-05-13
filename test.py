import numpy as np
import matplotlib.pyplot as plt
import random

# Simulation parameters
GRID_SIZE = 201            # Must be odd so there's a clear center
CENTER = GRID_SIZE // 2
NUM_PARTICLES = 5000
STICKING_PROBABILITY = 1.0  # 1.0 = always stick when adjacent

# Initialize the grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
grid[CENTER, CENTER] = 1  # Seed particle at center

def is_adjacent_to_cluster(x, y):
    """Check if the given cell is adjacent to an occupied cell"""
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if grid[nx, ny] == 1:
                    return True
    return False

def perform_random_walk():
    """Perform a random walk until the particle sticks or escapes the grid"""
    # Start particle at a random point near the border
    angle = random.uniform(0, 2 * np.pi)
    radius = CENTER - 2
    x = int(CENTER + radius * np.cos(angle))
    y = int(CENTER + radius * np.sin(angle))

    while 0 < x < GRID_SIZE-1 and 0 < y < GRID_SIZE-1:
        # Random walk (Brownian motion)
        dx, dy = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
        x += dx
        y += dy

        # Check for contact
        if is_adjacent_to_cluster(x, y):
            if random.random() < STICKING_PROBABILITY:
                grid[x, y] = 1
                return True
    return False

# Run the simulation
for i in range(NUM_PARTICLES):
    stuck = perform_random_walk()
    if i % 100 == 0:
        print(f"{i} particles released...")



# Plot the result
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap='inferno', origin='lower')
plt.title("Diffusion-Limited Aggregation (DLA) – Bacterial Growth Simulation")
plt.axis('off')
plt.show()

def box_counting(grid, min_box_size=2, max_box_size=100, step=2):
    """Compute box-counting fractal dimension for the grid"""
    sizes = []
    counts = []

    for box_size in range(min_box_size, max_box_size, step):
        num_boxes = 0
        for i in range(0, grid.shape[0], box_size):
            for j in range(0, grid.shape[1], box_size):
                sub_box = grid[i:i+box_size, j:j+box_size]
                if np.any(sub_box):
                    num_boxes += 1
        if num_boxes > 0:
            sizes.append(box_size)
            counts.append(num_boxes)

    sizes = np.array(sizes)
    counts = np.array(counts)
    
    log_sizes = np.log(1.0 / sizes)
    log_counts = np.log(counts)

    # Linear fit (slope gives the dimension)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dimension = coeffs[0]
    
    # Plot log-log
    plt.figure()
    plt.plot(log_sizes, log_counts, 'o-', label=f"Estimated D ≈ {fractal_dimension:.3f}")
    plt.xlabel("log(1/ε)")
    plt.ylabel("log N(ε)")
    plt.title("Box-Counting Fractal Dimension")
    plt.legend()
    plt.grid(True)
    plt.show()

    return fractal_dimension

# Run after simulation
dimension = box_counting(grid)
print(f"Estimated fractal dimension: {dimension:.3f}")
