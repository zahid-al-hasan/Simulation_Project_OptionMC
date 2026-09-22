"""Random and quasi-random number generators for Monte Carlo simulation."""

from abc import ABC, abstractmethod
import math

import numpy as np
from scipy.stats import norm, qmc


class Sampler(ABC):
    """Base interface for samplers that produce standard-normal draws."""

    def __init__(self, n_paths: int, seed: int | None = None):
        if not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
            raise ValueError("n_paths must be a positive integer")
        self.n_paths = int(n_paths)
        self.seed = seed

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Return standard-normal draws."""


class StandardNormalSampler(Sampler):
    """Generate independent standard-normal draws."""

    def __init__(self, n_paths: int, seed: int | None = None):
        super().__init__(n_paths, seed)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self.rng.standard_normal(self.n_paths)


class _QuasiRandomSampler(Sampler):
    """Shared functionality for low-discrepancy uniform-to-normal mapping."""

    def __init__(self, n_paths: int, seed: int | None = None, d: int = 1):
        super().__init__(n_paths, seed)
        if not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError("d must be a positive integer")
        self.d = int(d)

    @abstractmethod
    def uniform_sample(self) -> np.ndarray:
        """Return low-discrepancy points in the unit cube."""

    def sample(self) -> np.ndarray:
        uniform = self.uniform_sample()
        epsilon = np.finfo(float).eps
        normal = norm.ppf(np.clip(uniform, epsilon, 1.0 - epsilon))
        return normal[:, 0] if self.d == 1 else normal


class SobolSampler(_QuasiRandomSampler):
    """Generate scrambled Sobol points mapped to standard-normal draws."""

    def __init__(self, n_paths: int, seed: int | None = None, d: int = 1):
        super().__init__(n_paths, seed, d)
        self.sobol_engine = qmc.Sobol(d=self.d, scramble=True, seed=self.seed)

    def uniform_sample(self) -> np.ndarray:
        if self.n_paths & (self.n_paths - 1) == 0:
            exponent = int(math.log2(self.n_paths))
            return self.sobol_engine.random_base2(exponent)
        return self.sobol_engine.random(self.n_paths)


class HaltonSampler(_QuasiRandomSampler):
    """Generate scrambled Halton points mapped to standard-normal draws."""

    def __init__(self, n_paths: int, seed: int | None = None, d: int = 1):
        super().__init__(n_paths, seed, d)
        self.halton_engine = qmc.Halton(d=self.d, scramble=True, seed=self.seed)

    def uniform_sample(self) -> np.ndarray:
        return self.halton_engine.random(self.n_paths)
