import numpy as np
import sys
import pdb

from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
from numpy.random import uniform


def visualize_map(occupancy_map):
    plt.figure()
    plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o', s=3)
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):
    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):
    # initialize [x, y, theta] positions in world_frame for all particles

    """
    TODO : Add your code here
    """
    X_bar_init = []
    w0_vals = 1 / num_particles
    for i in range(0, num_particles):
        x = int(uniform(0, 800))
        y = int(uniform(0, 800))
        theta = uniform(-np.pi, np.pi)
        while occupancy_map[x, y] != 0:
            x = int(uniform(0, 800))
            y = int(uniform(0, 800))
            theta = uniform(-np.pi, np.pi)
        X_bar_init.append(np.array([y * 10, x * 10, theta, w0_vals]))
    return np.asarray(X_bar_init)


def plot_map(occupancy_map, X_bar):
    walls = []
    for i in range(0, occupancy_map.shape[0]):
        for j in range(0, occupancy_map.shape[1]):
            if occupancy_map[i, j] != 0:
                walls.append([j * 10, i * 10])
    walls = np.asarray(walls)
    X_bar = np.asarray(X_bar)
    plt.plot(walls[:, 0], walls[:, 1], 'bo', markersize=1)
    plt.plot(X_bar[:, 0], X_bar[:, 1], 'ro', markersize=1)
    plt.show()


def main():
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """

    """
    Initialize Parameters
    """
    src_path_map = '../data/map/wean.dat'
    src_path_log = '../data/log/robotdata2.log'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    params = {
        'z_max': 8183,
        'lambda_short': 0.01,
        'sigma_hit': 250,

        'z_pHit': 1000,
        'z_pShort': 0.01,
        'z_pMax': 0.03,
        'z_pRand': 100000,

        'laser_sensor_offset': 25.0,
        'ray_step_size': 2,
        'grid_size': 10,
        'occ_thrsh': 0.1,
        'laser_subsample': 30,

        'rayCast_vis': False,
        'map_vis': True
    }
    sensor_model = SensorModel(occupancy_map, params)
    resampler = Resampling()

    num_particles = 100
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    vis_flag = 1

    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if vis_flag:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0]  # L : laser scan measurement, O : odometry measurement
        meas_vals = np.fromstring(line[2:], dtype=np.float64,
                                  sep=' ')  # convert measurement values from string to double

        odometry_robot = meas_vals[0:3]  # odometry reading [x, y, theta] in odometry frame
        time_stamp = meas_vals[-1]

        # if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging) 
            # continue

        if (meas_type == "L"):
            odometry_laser = meas_vals[3:6]  # [x, y, theta] coordinates of laser in odometry frame
            ranges = meas_vals[6:-1]  # 180 range measurement values from single laser scan

        print("Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s")

        if (first_time_idx):
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot
        for m in range(0, num_particles):

            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                # w_t = 1/num_particles
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)
        print(len(X_bar))

        if vis_flag:
            visualize_timestep(X_bar)


if __name__ == "__main__":
    main()
