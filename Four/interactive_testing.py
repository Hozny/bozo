import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from main import TilingEnvironment 

# Configuration: global variables
GRID_SIZE = 10
NUM_POINTS = 10
NUM_SAMPLES = 10

# Create a list to store the points
points = []

def on_click(event):
    global points, NUM_POINTS

    # Add the clicked point to the list
    points.append([event.xdata, event.ydata])

    # Plot the point on the grid
    plt.plot(event.xdata, event.ydata, marker='o', color='r', markersize=5)

    # Update the plot
    plt.draw()

    # Check if the required number of points have been clicked
    if len(points) == NUM_POINTS:
        # Disconnect the click event
        plt.disconnect(cid)

        # Print the list of points
        print("Clicked points:", points)

        # Calculate and plot Voronoi regions
        vor = Voronoi(np.array(points))
        fig, ax = plt.subplots()
        voronoi_plot_2d(vor, ax=ax, show_vertices=False)
        ax.plot(np.array(points)[:, 0], np.array(points)[:, 1], 'r.')

        # Set fixed axis limits for the Voronoi plot
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)

        # Show the Voronoi plot
        plt.show()

        # Calculate and print the reward
        tile = TilingEnvironment()
        tile.points = points
        reward = tile._compute_reward()
        print("REWARD:", reward)

# Create the grid
fig, ax = plt.subplots()
plt.grid(True)

# Set fixed axis limits for the grid
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)

# Connect the click event
cid = plt.connect('button_press_event', on_click)

# Show the grid
plt.show()


def plot_voronoi(points, reward):
    # Calculate and plot Voronoi regions
    vor = Voronoi(np.array(points))
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False)
    ax.plot(np.array(points)[:, 0], np.array(points)[:, 1], 'r.')

    # Set fixed axis limits for the Voronoi plot
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    # Print the reward value
    print("REWARD:", reward)

    # Show the Voronoi plot
    plt.show()

# Generate random sets of points and compute their rewards
samples = []
for _ in range(NUM_SAMPLES):
    random_points = np.random.rand(NUM_POINTS, 2) * GRID_SIZE
    tile = TilingEnvironment()
    tile.points = random_points
    reward = tile._compute_reward()
    samples.append((random_points, reward))

# Sort the samples based on their reward values
samples.sort(key=lambda x: x[1])

# Display Voronoi regions and reward values for the top 3, median 3, and bottom 3 sets
for i in range(3):
    plot_voronoi(samples[i][0], samples[i][1])        # Bottom 3

    plot_voronoi(samples[-i-1][0], samples[-i-1][1])  # Top 3

median_index = len(samples) // 2
for i in range(3):
    plot_voronoi(samples[median_index+i-1][0], samples[median_index+i-1][1])  # Median 3
