# import pdb
import gym
from gym.spaces import Discrete, Box
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

        self.width = width
        self.height = height
        self.num_points = num_points
        self.points = self._get_default_grid(width, height, num_points)

        # Actions are a selection of points within the grid
        self.action_space = gym.spaces.Box(low=0, high=max(
            width, height), shape=(num_points, 2), dtype=np.float32)

        # For each point return its voronoi area and perimeter
        self.observation_space = gym.spaces.Box(low=0, high=max(
            width, height), shape=(num_points, 2), dtype=np.float32)

    # Return num_points evenly spaced in grid fashion within the grid of size
    # width x height. Ensures that all num_points are placed within the grid.

    def _get_default_grid(self, width, height, num_points):
        points = []
        x_step = width / num_points
        y_step = height / num_points
        for i in range(num_points):
            points.append([x_step * i, y_step * i])
        return np.array(points)

    def _take_action(self, action):
        self.points = action

        # Voronoi region of each point in action
        try:
            vor = Voronoi(action)
        except:
            dream = []
            for idx in range(len(self.points)):
                dream.append([BIG_INF, BIG_INF])
            return -1*BIG_INF, np.array(dream)

        # vor_regions: the actual point coordinates of the voronoi regions taken
        # from the points in vor.regions
        vor_regions = []
        for ii in vor.point_region:
            region = vor.regions[ii]
            def f(idx): return vor.vertices[idx]
            points = f(region)
            vor_regions.append(points)

        # for each set of points in vor_regions, calculate the area of the region
        # and append it to the end of the regions array. Claculate areas using shoelace algorithm
        # Could this be vectorized more?
        vor_perimeters = np.array([])
        vor_num_sides = np.array([])
        vor_areas = np.array([])

        adjusted_area_mean = np.array([])
        adjusted_perimeter_mean = np.array([])
        for region in vor_regions:
            x = region[:, 0]
            y = region[:, 1]
            area = 1
            area = np.abs(
                0.5 * np.array(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

            if len(region) == 0:
                area = BIG_INF
                perimeter = 2
            elif -1 in region:
                perimeter = BIG_INF
                area = BIG_INF
            else:
                # perimeter = np.sum([euclidean(a, b) for a, b in zip(x, y)])
                perimeter = 1
                adjusted_area_mean = np.append(adjusted_area_mean, area)
                adjusted_perimeter_mean = np.append(
                    adjusted_perimeter_mean, perimeter)

            vor_perimeters = np.append(vor_perimeters, perimeter)
            vor_areas = np.append(vor_areas, area)
            vor_num_sides = np.append(vor_num_sides, len(region))

        vor_region_obs = np.vstack((vor_areas, vor_perimeters)).T

        # NO big inf
        adjusted_area_mean = np.mean(adjusted_area_mean)
        adjusted_perimeter_mean = np.mean(adjusted_perimeter_mean)

        if len(vor_region_obs) < self.num_points:
            for i in range(self.num_points - len(vor_region_obs)):
                vor_region_obs = np.append(vor_region_obs, [BIG_INF, 1])

        area_rew = np.sum(
            (vor_areas - adjusted_area_mean)**2)
        perimeter_rew = np.sum(
            (vor_perimeters - adjusted_perimeter_mean)**2)
        ratio_rew = np.sum((vor_areas / vor_perimeters)**2)

        reward = -area_rew-perimeter_rew-ratio_rew

        print("REWARD: ", reward)
        return reward, vor_region_obs

    def step(self, action):
        reward, obs = self._take_action(action)
        return obs, reward, False, {}  # never done, no debug

    # We can consider setting points to a bunch of random points in the grid
    # Either each time or randomly in an episode
    def reset(self):
        self.points = self._get_default_grid(
            self.width, self.height, self.num_points)
        return self.points

    def render(self, mode='human', close=False):
        fig = voronoi_plot_2d(Voronoi(self.points))
        plt.show()
        print(self.points)
