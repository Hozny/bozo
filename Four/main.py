import argparse
import numpy as np
import gym
from gym import spaces
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


class TilingEnvironment(gym.Env):
    def __init__(self, grid_size=10, max_points=6, tiling_weight=0.5):
        super(TilingEnvironment, self).__init__()

        self.grid_size = grid_size
        self.max_points = max_points
        self.tiling_weight = tiling_weight

        self.action_space = spaces.Discrete(grid_size * grid_size)
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(max_points * 2,), dtype=np.float32)

        self.points = []
        self.reset()

    def reset(self):
        self.points = []
        return np.zeros((self.max_points * 2,), dtype=np.float32)

    def step(self, action):
        x = action // self.grid_size
        y = action % self.grid_size
        self.points.append((x, y))

        if len(self.points) < self.max_points:
            return self._get_observation(), 0, False, {}

        reward = self._compute_reward()
        self.reset()
        return self._get_observation(), reward, True, {}

    def _get_observation(self):
        obs = np.zeros((self.max_points * 2,), dtype=np.float32)
        for i, point in enumerate(self.points):
            obs[i * 2] = point[0]
            obs[i * 2 + 1] = point[1]
        return obs

    def _compute_tiling_similarity(self, polygon):
        total_similarity = 0
        n = len(self.points)
        num_regions = len(self.vor.regions)  # Add this line

        for i in range(n):
            for j in range(i + 1, n):
                if j >= num_regions:  # Add this condition
                    continue

                poly_i = self._get_voronoi_polygon(i)
                poly_j = self._get_voronoi_polygon(j)

                if poly_i is None or poly_j is None:
                    continue

                intersection_area = poly_i.intersection(poly_j).area
                union_area = poly_i.union(poly_j).area
                jaccard_similarity = intersection_area / union_area if union_area > 0 else 0
                total_similarity += jaccard_similarity
        return total_similarity / (n * (n - 1) / 2)

    def _get_voronoi_polygon(self, i):
        if -1 in self.vor.regions[i]:
            return None
        return Polygon([self.vor.vertices[v] for v in self.vor.regions[i]])

    def _compute_reward(self):
        self.vor = Voronoi(self.points, qhull_options="QJ")

        total_area = 0
        total_perimeter = 0
        total_tiling_similarity = 0

        for i, region in enumerate(self.vor.regions):
            if -1 in region or len(region) == 0:  # Skip unbounded regions
                continue

            polygon = self._get_voronoi_polygon(i)
            if polygon is None or not polygon.is_valid:
                continue

            total_area += polygon.area
            total_perimeter += polygon.length
            total_tiling_similarity += self._compute_tiling_similarity(polygon)

        area_perimeter_ratio = 0 if total_perimeter == 0 else total_area / total_perimeter
        tiling_similarity = total_tiling_similarity / len(self.vor.regions)
        return self.tiling_weight * tiling_similarity + (1 - self.tiling_weight) * area_perimeter_ratio

    def render(self, mode='human'):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)

        # Plot Voronoi diagram
        if len(self.points) >= 4:  # Change this line
            # Ensure self.vor exists
            if not hasattr(self, 'vor'):
                self.vor = Voronoi(self.points, qhull_options="QJ")
            voronoi_plot_2d(self.vor, ax=ax, show_vertices=False,
                            line_colors='orange', line_width=1.5, line_alpha=0.6, point_size=2)

        # Plot points
        for point in self.points:
            ax.scatter(point[0], point[1], color='blue', s=50)

        plt.grid(True)
        plt.gca().set_xticks(np.arange(0, self.grid_size + 1, 1))
        plt.gca().set_yticks(np.arange(0, self.grid_size + 1, 1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train or test a Voronoi Tiling agent using PPO.")
parser.add_argument("--load", action="store_true", help="Load from the saved model instead of retraining.")
parser.add_argument("--grid_size", type=int, default=10, help="Number of rows and columns in the grid (default: 10).")
parser.add_argument("--max_points", type=int, default=6, help="Maximum number of Voronoi points (default: 6).")
args = parser.parse_args()

# Create the environment
env = TilingEnvironment(grid_size=args.grid_size, max_points=args.max_points)

if args.load:
    # Load the saved model
    model = PPO.load("ppo_tiling_agent")
else:
    # Train the agent using PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    # Save the model
    model.save("ppo_tiling_agent")

# Test the trained agent
observation = env.reset()
done = False
while not done:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)
    env.render()
