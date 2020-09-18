import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
import pdb
from numpy import sin, cos

from MapReader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map, params):

        """
        - Initialize the map here
        - Initialize the measurements here
        """
        self.occupancy_map = occupancy_map

        self.laser_sensor_offset = params['laser_sensor_offset']
        self.ray_step_size = params['ray_step_size']  # Distance (in cm) used for forwarding procedure in ray-tracing
        self.grid_size = params['grid_size']  # Occupancy grid resolution (Parameter required for ray-tracing)
        self.thrsh = params['occ_thrsh']  # Threshold value required to check for occupancy/collision
        self.laser_subsample_factor = params['laser_subsample']

        self.z_max = params['z_max']
        self.var_hit = params['sigma_hit']
        self.lambda_short = params['lambda_short']

        self.z_pHit = params['z_pHit']
        self.z_pShort = params['z_pShort']
        self.z_pMax = params['z_pMax']
        self.z_pRand = params['z_pRand']

        self.rayCast_vis = params['rayCast_vis']

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        # Algorithm Table 6.1
        # Generate the ray cast solution to retrieve z_true
        x, y, theta = x_t1

        # safety check
        sft_chk = self.occupancy_map[min(int(y / self.grid_size), self.occupancy_map.shape[0] - 1)] \
            [min(int(x / self.grid_size), self.occupancy_map.shape[1] - 1)]
        if sft_chk > 0.4 or sft_chk == -1:
            return 1e-100

        q = 0

        laser_x, laser_y = self.laser_sensor_offset * np.cos(theta), self.laser_sensor_offset * np.sin(theta)

        map_idx_x, map_idx_y = int(round((x + laser_x) / self.grid_size)), int(round((y + laser_y) / self.grid_size))

        for samples_deg in range(-90, 90, self.laser_subsample_factor):

            true_range = self.rayCast(samples_deg, theta, map_idx_x, map_idx_y)

            measure = z_t1_arr[samples_deg + 90]

            # Line 5 of Algorithm Table 6.1
            p1 = self.z_pHit * self.p_hit(measure, x_t1, true_range)
            p2 = self.z_pShort * self.p_short(measure, x_t1, true_range)
            p3 = self.z_pMax * self.p_max(measure, x_t1)
            p4 = self.z_pRand * self.p_rand(measure, x_t1)

            p = p1 + p2 + p3 + p4
            if p > 0:
                q = q + np.log(p)

        return math.exp(q)

    def rayCast(self, samples_deg, theta, map_idx_x, map_idx_y):

        final_angle = theta + math.radians(samples_deg)

        start_x, start_y = map_idx_x, map_idx_y

        final_x, final_y = map_idx_x, map_idx_y

        while 0 < final_x < self.occupancy_map.shape[1] and 0 < final_y < self.occupancy_map.shape[0] \
                and abs(self.occupancy_map[final_y, final_x]) < 0.0000001:
            start_x += self.ray_step_size * np.cos(final_angle)
            start_y += self.ray_step_size * np.sin(final_angle)

            final_x, final_y = int(round(start_x)), int(round(start_y))

        final = np.array([final_x, final_y])

        init = np.array([map_idx_x, map_idx_y])
        dist = np.linalg.norm(final - init) * 10
        return dist

    def p_hit(self, z_tk, x_t, z_tk_star):
        if 0 <= z_tk <= self.z_max:
            gaussian = (math.exp(-(z_tk - z_tk_star) ** 2 / (2 * self.var_hit ** 2))) / math.sqrt(
                2 * math.pi * self.var_hit ** 2)
            return gaussian
        else:
            return 0.0

    def p_short(self, z_tk, x_t, z_tk_star):
        if 0 <= z_tk <= z_tk_star:
            eta = 1 / (1 - math.exp(-self.lambda_short * z_tk_star))
            return eta * self.lambda_short * math.exp(-self.lambda_short * z_tk)
        else:
            return 0.0

    def p_max(self, z_tk, x_t):
        if z_tk == self.z_max:
            return 1.0
        else:
            return 0.0

    def p_rand(self, z_tk, x_t):
        if 0 <= z_tk < self.z_max:
            return 1.0 / self.z_max
        else:
            return 0.0

    def collision_check(self, occupancy_map, succeeded_distance_x, succeeded_distance_y):
        succeeded_distance_index_x = int(succeeded_distance_x // self.grid_size)
        succeeded_distance_index_y = int(succeeded_distance_y // self.grid_size)

        if (succeeded_distance_index_x < 0 or succeeded_distance_index_x >= occupancy_map.shape[1]
                or succeeded_distance_index_y < 0 or succeeded_distance_index_y >= occupancy_map.shape[0]):
            return False
        elif occupancy_map[succeeded_distance_index_y, succeeded_distance_index_x] < self.thrsh:
            return False
        else:
            return True

    def plot_rayCast(self, succeeded_distance_x, succeeded_distance_y, ray_directions, z_true):
        """ Show the ray cast result. Ray in red color means obstacle being detected
            within the laser max range; in blue color means fail to detect obstacle.
        Args:
          succeeded_distance: the location of the laser location in world coordinates, cm
          ray_directions: list of the directions of the laser rays, radians
          z_true: list of true ranges between the laser and the obstacles in
                       directions listed in ray_directions.
        Returns:
        """
        plt.ion()
        plt.imshow(self.occupancy_map, cmap='Greys', origin='lower')
        plt.scatter(succeeded_distance_x / 10, succeeded_distance_y / 10, c='r', marker='.')

        x_start, y_start = succeeded_distance_x / 10, succeeded_distance_y / 10
        for index, ray_direction in enumerate(ray_directions):
            fails, range = z_true[index]
            x_dest = x_start + range * math.cos(ray_direction) / 10
            y_dest = y_start + range * math.sin(ray_direction) / 10
            color = 'b' if fails else 'r'
            plt.plot([x_start, x_dest], [y_start, y_dest], c=color, linewidth=1)
        plt.pause(0)

    def get_pointcloud(self, X, scan):
        rot = np.arange(-np.pi / 2, np.pi / 2, np.pi / 180)
        scan_points = []
        for i in range(0, len(scan)):
            ang = X[2] + rot[i]
            x = X[0] + scan[i] * cos(ang)
            y = X[1] + scan[i] * sin(ang)
            scan_points.append([x, y])
        return np.asarray(scan_points)


if __name__ == '__main__':
    src_path_map = '../data/map/wean.dat'

    # Important stuff play with parameters
    params = {
        'z_max': 8000,
        'lambda_short': 0.1,
        'sigma_hit': 2,

        'z_pHit': 0.95,
        'z_pShort': 0.01,
        'z_pMax': 0.05,
        'z_pRand': 0.05,

        'laser_sensor_offset': 25.0,
        'ray_step_size': 2,
        'grid_size': 10,
        'occ_thrsh': 0.1,
        'laser_subsample': 30,

        'rayCast_vis': False,
        'map_vis': True
    }
    map1 = MapReader(src_path_map)
    occupancy_map = map1.get_map()
    model = SensorModel(occupancy_map, params)

    walls = []
    for i in range(0, occupancy_map.shape[0]):
        for j in range(0, occupancy_map.shape[1]):
            if occupancy_map[i, j] != 0:
                walls.append([j * 10, i * 10])
    walls = np.asarray(walls)
    plt.plot(walls[:, 0], walls[:, 1], 'bo', markersize=1)

    X = np.array([4610, 2190, 0])
    plt.plot(X[0], X[1], 'go')
    rays = []
    for i in range(-90, 90):
        rays.append(model.rayCast(i, X[2], X[0] // 10, X[1] // 10))
    pc = model.get_pointcloud(X, rays)
    plt.plot(pc[:, 0], pc[:, 1], 'ro', markersize=2)
    plt.show()
