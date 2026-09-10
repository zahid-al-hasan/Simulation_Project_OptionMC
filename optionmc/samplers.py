"""Random and quasi-random number generators for Monte Carlo simulation."""

import numpy as np
from scipy.stats import qmc, norm


class Sampler:
    def __init__(self, n_paths, seed=None):
        self.n_paths = n_paths
        self.seed = seed
        pass

    def sample(self):
        pass


class StandardNormalSampler(Sampler):
    """Generate i.i.d. standard normal draws."""

    def __init__(self, n_paths, seed=None):
        super().__init__(n_paths, seed)
        self.rng = np.random.default_rng(seed=self.seed)

    def sample(self):
        # TODO: return standard normal array of shape (n_paths,)
        z_array = self.rng.standard_normal(size=self.n_paths)
        return z_array
        pass


class SobolSampler(Sampler):
    """Generate quasi-random Sobol sequences mapped to standard normal."""

    def __init__(self, n_paths, seed=None, d=1):
        super().__init__(n_paths, seed)
        self.d = d
        # TODO: set up Sobol engine via scipy.stats.qmc.Sobol
        self.sobol_engine = qmc.Sobol(d=self.d, scramble=True, seed=self.seed)

    def sample(self):
        # TODO: generate Sobol points in [0,1], map to standard normal via inverse CDF
        u = self.sobol_engine.random(self.n_paths)    
        return norm.ppf(u.flatten())
        pass


class HaltonSampler(Sampler):
    """Generate quasi-random Halton sequences mapped to standard normal."""

    def __init__(self, n_paths, seed=None, d=1):
        super().__init__(n_paths, seed)
        self.d = d
        # TODO: set up Halton engine via scipy.stats.qmc.Halton
        self.halton_engine = qmc.Halton(d=self.d, scramble=True, seed=self.seed)

    def sample(self):
        # TODO: generate Halton points in [0,1], map to standard normal via inverse CDF
        u = self.halton_engine.random(self.n_paths)
        return norm.ppf(u.flatten())
        pass
