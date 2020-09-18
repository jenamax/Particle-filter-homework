import numpy as np
import random
class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        self.trheshold_believe = 0.5

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        # X_bar_resampled = resampler.multinomial_sampler(X_bar)
        self.normalization_(X_bar)
        num_particles = np.shape(X_bar)[0]
        X_bar_resampled = np.zeros([num_particles, 4])

        for i in range(num_particles):
            x_bar_index = np.random.choice(num_particles, p=X_bar[:, 3])
            X_bar_resampled[i, :] = X_bar[x_bar_index, :]

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        X_bar_resampled = []
        M = len(X_bar)
        wt = X_bar[:,3]
        r = random.uniform(0, 1.0/M)
        wt = self.normalization_(wt)
        c = wt[0]
        i = 0
        for m in range(M):
            u = r + (m)*(1.0/M)
            while u>c:
                i = i +1
                c = c + wt[i]
            X_bar_resampled.append(X_bar[i])
        X_bar_resampled = np.asarray(X_bar_resampled)

        return X_bar_resampled

    def normalization_(self, weights):

        normalizer = np.sum(weights)
        if normalizer < self.trheshold_believe:
          print ('Total belief on the observed data is less than {0}.'.format(self.trheshold_believe))
        return weights / normalizer


if __name__ == "__main__":
    pass
