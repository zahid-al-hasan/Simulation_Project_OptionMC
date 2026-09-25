"""Variance-reduction techniques for Monte Carlo option pricing."""

import numpy as np

from optionmc.samplers import HaltonSampler, SobolSampler, StratifiedSampling


class AntitheticVariates:
    """Pair each standard-normal draw with its negation."""

    @staticmethod
    def transform(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        draws = np.asarray(Z, dtype=float)
        if draws.ndim != 1:
            raise ValueError("Z must be a one-dimensional array")
        return draws, -draws


class ControlVariates:
    """Use discounted terminal stock value as a known-mean control."""

    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)

    @staticmethod
    def _paired_arrays(
        payoffs: np.ndarray, control_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        payoff_array = np.asarray(payoffs, dtype=float)
        control_array = np.asarray(control_values, dtype=float)
        if payoff_array.ndim != 1 or control_array.ndim != 1:
            raise ValueError("payoffs and control_values must be one-dimensional")
        if payoff_array.size != control_array.size or payoff_array.size < 2:
            raise ValueError("payoffs and control_values must have equal length >= 2")
        return payoff_array, control_array

    def optimal_coefficient(
        self, payoffs: np.ndarray, control_values: np.ndarray
    ) -> float:
        payoff_array, control_array = self._paired_arrays(payoffs, control_values)
        control_variance = np.var(control_array, ddof=1)
        if control_variance <= 0.0:
            raise ValueError("control_values must have nonzero sample variance")
        covariance = np.cov(payoff_array, control_array, ddof=1)[0, 1]
        return float(covariance / control_variance)

    def adjusted_estimates(
        self, payoffs: np.ndarray, control_values: np.ndarray, beta: float
    ) -> np.ndarray:
        payoff_array, control_array = self._paired_arrays(payoffs, control_values)
        return payoff_array - float(beta) * (control_array - self.S0)


class QuasiMonteCarlo:
    """Generate terminal stock prices with Sobol or Halton draws."""

    def __init__(self, n_paths: int, method: str = "sobol", seed: int | None = None):
        if not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
            raise ValueError("n_paths must be a positive integer")
        if method not in {"sobol", "halton"}:
            raise ValueError("method must be 'sobol' or 'halton'")
        self.n_paths = n_paths
        self.method = method
        self.seed = seed

    def generate_paths(self, gbm_model) -> np.ndarray:
        sampler_class = SobolSampler if self.method == "sobol" else HaltonSampler
        return gbm_model.simulate(sampler_class(self.n_paths, seed=self.seed))
