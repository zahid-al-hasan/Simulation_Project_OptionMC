"""Variance reduction techniques for Monte Carlo option pricing."""

import numpy as np


class AntitheticVariates:
    """Antithetic variates: pair each draw with its negation."""

    @staticmethod
    def transform(Z):
        """Given Z ~ N(0,1) of shape (n/2,), return (Z, -Z) stacked -> (n,).

        Returns
        -------
        Z_original, Z_antithetic : tuple of np.ndarray
        """
        # TODO: return Z and -Z
        pass


class ControlVariates:
    """Control variates: use a known-average quantity to reduce variance.

    For European options, a natural control is the discounted stock price
    E[S_T * exp(-rT)] = S0 (known analytically).
    """

    def __init__(self, S0, K, r, sigma, T):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def optimal_coefficient(self, payoffs, control_values):
        """Compute optimal beta = Cov(payoff, control) / Var(control).

        Parameters
        ----------
        payoffs : np.ndarray – option payoff samples
        control_values : np.ndarray – control variate samples

        Returns
        -------
        float – optimal control coefficient beta
        """
        # TODO: compute optimal beta
        pass

    def adjusted_estimates(self, payoffs, control_values, beta):
        """Return variance-reduced payoff estimates.

        adjusted = payoffs - beta * (control_values - E[control])

        Parameters
        ----------
        payoffs : np.ndarray
        control_values : np.ndarray
        beta : float

        Returns
        -------
        np.ndarray – adjusted payoff samples
        """
        # TODO: compute adjusted payoffs
        pass


class StratifiedSampling:
    """Stratified sampling: partition [0,1] into strata, sample within each."""

    def __init__(self, n_strata, n_samples_per_stratum):
        self.n_strata = n_strata
        self.n_samples_per_stratum = n_samples_per_stratum

    def stratified_uniform(self):
        """Generate stratified uniform samples on [0,1].

        Returns
        -------
        np.ndarray – stratified samples of shape (n_strata * n_samples_per_stratum,)
        """
        # TODO: partition [0,1] into n_strata intervals, draw uniformly within each
        pass


class QuasiMonteCarlo:
    """Quasi-Monte Carlo using low-discrepancy sequences instead of PRNG."""

    def __init__(self, n_paths, method="sobol", seed=None):
        self.n_paths = n_paths
        self.method = method
        self.seed = seed

    def generate_paths(self, gbm_model):
        """Generate terminal stock prices using quasi-random sequences.

        Parameters
        ----------
        gbm_model : GeometricBrownianMotion

        Returns
        -------
        np.ndarray – terminal stock prices
        """
        # TODO: use Sobol/Halton to generate Z, then pass to GBM
        pass
