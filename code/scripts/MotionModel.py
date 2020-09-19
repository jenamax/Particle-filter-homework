import sys
import numpy as np
import math

class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):

        """
        Initializing of Motion Model parameters here
        """

        self.a1 = 0.0001
        self.a2 = 0.0001
        self.a3 = 0.1
        self.a4 = 0.1

    def update(self, u_t0, u_t1, x_t0):

        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        """
        In the book, 0.1s proposed as delta_t
        """

        x = x_t0[0]
        y = x_t0[1]
        theta = x_t0[2]

        # Here the odom indicates that these are odometry measurements,
        # embedded in a robot-internal coordinate whose relation to the global world coordinates
        # is unknown.

        x_odom = u_t0[0]
        y_odom = u_t0[1]
        theta_odom = u_t0[2]

        x_odom_ = u_t1[0]
        y_odom_ = u_t1[1]
        theta_odom_ = u_t1[2]

        rot1 = math.atan2(y_odom_ - y_odom,x_odom_ - x_odom) - theta_odom
        trasi = math.sqrt((x_odom-x_odom_)**2 + (y_odom-y_odom_)**2)
        rot2 = theta_odom_ - theta_odom - rot1

        rot1_ = rot1 - np.random.normal(0, np.sqrt((self.a1*(rot1**2)) + (self.a2*(trasi**2))))
        trasi_ = trasi - np.random.normal(0,np.sqrt((self.a3*(trasi**2)) + (self.a4*(rot1**2)) + (self.a4*(rot2**2))))
        rot2_ = rot2 - np.random.normal(0,np.sqrt((self.a1*(rot2**2)) + (self.a2*(trasi**2))))


        x_ = x + trasi_*math.cos(x_t0[2] + rot1_)
        y_ = y + trasi_*math.sin(x_t0[2] + rot1_)
        theta_ = theta + rot1_ + rot2_


        x_t1 = np.array([x_, y_, theta_])
        return x_t1
    # function sample(b) generates a random sample from a zero-centered distribution with variance b
    def sample(self, b, algorithm="normal"):
        if algorithm == "normal":
            value = b/6
            for i in range(12):
                value = value * self.rand(-1, 1)
            return value
        else:
            return b * self.rand(-1, 1)*self.rand(-1, 1)

    def rand(self, x, y):
        return np.random.uniform(x, y)
