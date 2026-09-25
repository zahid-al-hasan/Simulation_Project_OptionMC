"""Monte Carlo pricing engine for European options."""

import numpy as np
import time
from samplers import Sampler, SobolSampler, HaltonSampler, StandardNormalSampler
from models import GeometricBrownianMotion
from config import *

class OptionPricing:
    """Core MC pricing engine supporting multiple variance reduction methods."""

    def __init__(self, S0, K, r, sigma, T, n_paths, seed=None):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_paths = n_paths
        self.seed = seed

    def standard_mc(self, sampler: Sampler, model: GeometricBrownianMotion, option_type="call"):
        """Plain Monte Carlo pricing.

        Returns
        -------
        dict with keys: price, std_error, ci_lower, ci_upper, runtime
        """
        # TODO: generate Z, simulate S_T, compute payoffs, discount, return stats
        start_time = time.time()
        Z = sampler.sample()
        S_T = model.simulate(sampler=sampler)
        end_time = time.time()
        payoff = max(S_T - self.K, 0)
        if option_type == "put":
            payoff = max(self.K - S_T, 0)

        discount = np.exp(-self.r * (self.T))
        price = discount * np.mean(payoff)
        std_err = np.std(payoff) / np.sqrt(sampler.n_paths)

        return {
            "price" : price, 
            "std_err" : std_err,
            "ci_lower" : price - Z_REF * std_err,
            "ci_upper" : price + Z_REF * std_err,
            "runtime" : end_time - start_time
        }
        pass

    def antithetic_mc(self, option_type="call"):
        """MC with antithetic variates.

        Returns
        -------
        dict with keys: price, std_error, ci_lower, ci_upper, runtime
        """
        # TODO: use AntitheticVariates to generate paths, price, return stats
        pass

    def control_variate_mc(self, option_type="call"):
        """MC with control variates (control = discounted stock price).

        Returns
        -------
        dict with keys: price, std_error, ci_lower, ci_upper, runtime
        """
        # TODO: implement control variate pricing
        pass

    def stratified_mc(self, option_type="call", n_strata=10):
        """MC with stratified sampling.

        Returns
        -------
        dict with keys: price, std_error, ci_lower, ci_upper, runtime
        """
        # TODO: implement stratified MC pricing
        pass

    def quasi_mc(self, option_type="call", method="sobol"):
        """Quasi-Monte Carlo using low-discrepancy sequences.

        Returns
        -------
        dict with keys: price, std_error, ci_lower, ci_upper, runtime
        """
        # TODO: implement QMC pricing
        pass

    def confidence_interval(self, mean, std_error, confidence=0.95):
        """Compute confidence interval for MC estimate."""
        # TODO: use scipy.stats.norm.ppf or z-score
        pass
