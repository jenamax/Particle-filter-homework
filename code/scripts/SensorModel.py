import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
import pdb

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

        self.laser_sensor_offset = params['laser_sensor_offset']  # From Piazza
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

        """ Sebastian Thurn, Probablistic Robotics -- Line 4 of Algorithm Table 6.1 """
        """ Generate the ray cast solution to retrieve z_true """
        x, y, theta = x_t1

        """ Safety check """
        sft_chk = self.occupancy_map[min(int(y / self.grid_size), self.occupancy_map.shape[0] - 1)] \
            [min(int(x / self.grid_size), self.occupancy_map.shape[1] - 1)]
        if sft_chk > 0.4 or sft_chk == -1:
            return 1e-100

        """ rayCast visualization using legacy code """
        if self.rayCast_vis:
            measurements = z_t1_arr[::self.laser_subsample_factor]
            z_true = self.rayCast_legacy(measurements, x_t1, self.occupancy_map)

        q = 0

        laser_x, laser_y = self.laser_sensor_offset * np.cos(theta), self.laser_sensor_offset * np.sin(theta)

        map_idx_x, map_idx_y = int(round((x + laser_x) / self.grid_size)), int(round((y + laser_y) / self.grid_size))

        for samples_deg in range(-90, 90, self.laser_subsample_factor):

            true_range = self.rayCast(samples_deg, theta, map_idx_x, map_idx_y)

            measure = z_t1_arr[samples_deg + 90]

            """ Generate the 4 probability distributions -- Line 5 of Algorithm Table 6.1"""
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

    def rayCast_legacy(self, measurements, x_t1, occupancy_map):
        """
        param[in] measurements : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[in] occupancy_map: map of the environment
        param[out] z_true: true laser range readings retrieved using ray tracing
        """

        """ x, y  -- location of laser in cm """
        """ theta -- orientation of laser in radians """
        x, y, theta = x_t1
        laser_coord_x, laser_coord_y = x + (self.laser_sensor_offset) * np.cos(theta), y + (
            self.laser_sensor_offset) * np.sin(theta)

        """ Generate samples with which to carry out ray tracing """
        rays_counterclockwise = list(np.arange(0, 180, self.laser_subsample_factor) * np.pi / 180 - np.pi / 2 + theta)

        if len(rays_counterclockwise) != len(measurements):
            print('Can not match the number of measurement and true range.')
            exit(1)

        """ Carry out ray tracing """
        z_true = []

        for idx, heading in enumerate(rays_counterclockwise):
            succeeded_distance_x, succeeded_distance_y = np.copy(laser_coord_x), np.copy(laser_coord_y)

            true_range = 0

            ray_succeeded = np.array([self.ray_step_size * np.cos(heading),
                                      self.ray_step_size * np.sin(
                                          heading)])  # a pair of unit proceedings in x and y directions

            while true_range <= self.z_max and not self.collision_check(occupancy_map, \
                                                                        succeeded_distance_x, \
                                                                        succeeded_distance_y):
                succeeded_distance_x += ray_succeeded[0]
                succeeded_distance_y += ray_succeeded[1]
                true_range += self.ray_step_size
            fails = True if true_range > self.z_max else False
            z_true.append((fails, true_range))
        self.plot_rayCast(laser_coord_x, laser_coord_y, rays_counterclockwise, z_true)

        return z_true

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


if __name__ == '__main__':
    pass