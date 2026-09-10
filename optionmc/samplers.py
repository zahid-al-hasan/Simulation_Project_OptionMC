"""Random and quasi-random number generators for Monte Carlo simulation."""

import numpy as np
from scipy.stats import qmc


class StandardNormalSampler:
    """Generate i.i.d. standard normal draws."""

    def __init__(self, n_paths, seed=None):
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def sample(self):
        # TODO: return standard normal array of shape (n_paths,)
        pass


class SobolSampler:
    """Generate quasi-random Sobol sequences mapped to standard normal."""

    def __init__(self, n_paths, d=1, seed=None):
        self.n_paths = n_paths
        self.d = d
        # TODO: set up Sobol engine via scipy.stats.qmc.Sobol

    def sample(self):
        # TODO: generate Sobol points in [0,1], map to standard normal via inverse CDF
        pass


class HaltonSampler:
    """Generate quasi-random Halton sequences mapped to standard normal."""

    def __init__(self, n_paths, d=1, seed=None):
        self.n_paths = n_paths
        self.d = d
        # TODO: set up Halton engine via scipy.stats.qmc.Halton

    def sample(self):
        # TODO: generate Halton points in [0,1], map to standard normal via inverse CDF
        pass
