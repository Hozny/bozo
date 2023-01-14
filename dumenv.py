import pdb
import gym
from gym.spaces import Discrete, Box, Sequence
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

import numpy as np

BIG_INF = 1e5
MAX_POINTS_PER_REGION = 10


'''
Possible ideas for reward function that promotes tiling shapes.
By default voronoi regions will tile the space, but we want to encourage
regions to be identical shapes. We can do this by penalizing regions that
are not the same shape as the average region.P
'''


class Voronoi2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, num_points):
        super(Voronoi2DEnv, self).__init__()

        self.width = 10
        self.height = 10
        self.num_points = 4
        self.points = self._get_default_grid(width, height, num_points)

        self.action_space = gym.spaces.Box(low=0, high=max(
            width, height), shape=(num_points, 2), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=0, high=max(
            width, height), shape=(num_points, 2), dtype=np.float32)

    # Return num_points evenly spaced in grid fashion within the grid of size
    # width x height. Ensures that all num_points are placed within the grid.

    def _get_default_grid(self, width, height, num_points):
        return np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

    def step(self, action):
        random_obs = []
        for i in range(self.num_points):
            random_obs.append(np.random.uniform(0, 10, 2))

        reward = np.random.uniform(0, 10)
        return random_obs, reward, False, {}  # never done, no debug

    # We can consider setting points to a bunch of random points in the grid
    # Either each time or randomly in an episode
    def reset(self):
        self.points = self._get_default_grid(
            self.width, self.height, self.num_points)
        return self.points

    def render(self, mode='human', close=False):
        # fig = voronoi_plot_2d(self.points)
        # plt.show()
        print(self.points)
