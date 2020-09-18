import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.stats import norm
import pdb
from numpy import sin, cos

import bresenham

from MapReader import MapReader


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_raycast(ray):
    x_locs = [x[0] for x in ray]
    y_locs = [x[1] for x in ray]
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.pause(0.1)
    scat.remove()


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):

        self.occupancy_map = occupancy_map
        self.deg_2_rad = np.pi / 180
        self.zmax = 8191
        self.lambda_short = 0.001
        self.sigma_sq_hit = 700

        self.z_hit = 0.8
        self.z_short = 0.133
        self.z_max = 0.000325
        self.z_rand = 0.133

    def get_point_cloud(self, X, scan):
        rot = np.arange(-np.pi / 2, np.pi / 2, np.pi / 180)
        scan_points = []
        for i in range(0, len(scan)):
            ang = X[2] + rot[i] 
            x = X[0] + scan[i] * cos(ang)
            y = X[1] + scan[i] * sin(ang) 
            scan_points.append([x, y])
        return scan_points

    def ray_cast(self, pos, angle):
        angle *= self.deg_2_rad
        theta = pos[2] + angle
        x = (pos[0] / 10)
        y = (pos[1] / 10)

        x += (25 * np.cos(pos[2]))
        x = int(x)
        y += (25 * np.sin(pos[2]))
        y = int(y)

        x_initial, y_initial = x, y

        max_dist = np.max(self.occupancy_map.shape)

        stride = 5
        dist = self.zmax
        try:
            self.occupancy_map[y, x] != -1
        except:
            pdb.set_trace()
        while self.occupancy_map[y, x] != -1:
            if self.occupancy_map[y, x] > 0.95:
                dist = min(((x - x_initial) ** 2 + (y - y_initial) ** 2) ** 0.5, max_dist)
                break

            x += stride * np.cos(theta)
            y += stride * np.sin(theta)
            x = int(x)
            y = int(y)

        vis_flag = 0
        if vis_flag:
            finalRayPoints = list(bresenham.bresenham(int(x_initial), int(y_initial), int(x), int(y)))
            visualize_raycast(finalRayPoints)

        return dist

        self.zmax = 100
        self.lambda_short = 10
        self.sigma_sq_hit = 10

    def get_Nu(self, z_tk, z_tk_star):
        return 1.0 / math.sqrt(2 * math.pi * self.sigma_sq_hit) * math.exp(
            ((z_tk - z_tk_star) ** 2) / (-2.0 * self.sigma_sq_hit))

    def get_p_hit(self, z_tk, z_tk_star):

        if 0 <= z_tk <= self.zmax:
            Nu = self.get_Nu(z_tk, z_tk_star)

            eta = 1.0 / (1e-11 + integrate.quad(lambda x: self.get_Nu(x, z_tk_star), 0, self.zmax)[0])
            return eta * Nu
        else:
            return 0.0

    def get_p_short(self, z_tk, z_tk_star):
        if 0 <= z_tk <= z_tk_star:
            eta = 1.0 / (1.0 - math.exp(-1.0 * self.lambda_short * z_tk_star))
            return eta * (self.lambda_short) * math.exp(-1.0 * self.lambda_short * z_tk)
        else:
            return 0.0

    def get_p_max(self, z_tk):
        if z_tk == self.zmax:
            return 1.0
        else:
            return 0.0

    def get_p_rand(self, z_tk):
        if 0 <= z_tk < self.zmax:
            return 1.0 / self.zmax
        else:
            return 0.0

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        q = 0

        if self.occupancy_map[int(x_t1[1] / 10), int(x_t1[0] / 10)] == -1:
            return 0

        for k in range(0, 180, 5):
            z_tk = z_t1_arr[k]
            z_tk_star = self.ray_cast(x_t1, k)
            p_hit = self.get_p_hit(z_tk, z_tk_star)
            p_short = self.get_p_short(z_tk, z_tk_star)
            p_max = self.get_p_max(z_tk)
            p_rand = self.get_p_rand(z_tk)

            p = self.z_hit * p_hit + self.z_short * p_short + self.z_max * p_max + self.z_rand * p_rand
            q += math.log(p, 10)

        return q


if __name__ == '__main__':
    src_path_map = '../data/map/wean.dat'
    src_path_log = '../data/log/robotdata1.log'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    sensor_model = SensorModel(occupancy_map)
    vis_flag = 0

    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if vis_flag:
        visualize_map(occupancy_map)

    first_time_idx = True
    X_bar = np.array([4600, 2030, 0])
    scan_points = [np.array([0, 0])]

    walls = []
    for i in range(0, sensor_model.occupancy_map.shape[0]):
        for j in range(0, sensor_model.occupancy_map.shape[1]):
            if sensor_model.occupancy_map[i, j] != 0:
                walls.append([j * 10 , i * 10])
    walls = np.asarray(walls)

    #plt.show()
 
    axes = plt.gca()
    axes.set_xlim(0, 8000)
    axes.set_ylim(0, 8000)
    
    plt.plot(X_bar[0], X_bar[1], 'go', markersize=3)
    plt.plot(np.asarray(walls)[:, 0], np.asarray(walls)[:, 1], 'bo', markersize=1)

    x_t1 = X_bar
    z_s = []
    for i in range(-90, 90):
        z_s.append(sensor_model.ray_cast(x_t1, i))
    pc = sensor_model.get_point_cloud(X_bar, z_s)
    plt.plot(np.asarray(pc)[:, 0], np.asarray(pc)[:, 1], 'ro', markersize=2)
    plt.show()

