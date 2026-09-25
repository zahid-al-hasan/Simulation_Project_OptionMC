"""Monte Carlo pricing engine for European options."""

from time import perf_counter
import warnings

import numpy as np
from scipy.stats import norm

from optionmc.models import GeometricBrownianMotion
from optionmc.samplers import HaltonSampler, SobolSampler, StandardNormalSampler
from optionmc.variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    StratifiedSampling,
)


class OptionPricing:
    """Price European calls and puts with Monte Carlo methods."""

    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float, n_paths: int, seed: int | None = None):
        if K <= 0:
            raise ValueError("K must be positive")
        if not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
            raise ValueError("n_paths must be an integer of at least 2")
        self.model = GeometricBrownianMotion(S0, r, sigma, T)
        self.S0 = self.model.S0
        self.K = float(K)
        self.r = self.model.r
        self.sigma = self.model.sigma
        self.T = self.model.T
        self.n_paths = int(n_paths)
        self.seed = seed


    @staticmethod
    def _validate_option_type(option_type: str) -> str:
        normalized = option_type.lower()
        if normalized not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        return normalized


    def _discounted_payoffs(self, terminal_prices: np.ndarray, option_type: str) -> np.ndarray:
        normalized = self._validate_option_type(option_type)
        prices = np.asarray(terminal_prices, dtype=float)
        if normalized == "call":
            payoff = np.maximum(prices - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - prices, 0.0)
        return np.exp(-self.r * self.T) * payoff


    def confidence_interval(self, mean: float, std_error: float, confidence: float = 0.95) -> tuple[float, float]:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if std_error < 0:
            raise ValueError("std_error cannot be negative")
        critical_value = norm.ppf(0.5 + confidence / 2.0)
        margin = critical_value * std_error
        return float(mean - margin), float(mean + margin)


    def _result(self, estimates: np.ndarray, start_time: float, method: str, n_paths: int | None = None, extra: dict | None = None) -> dict:
        values = np.asarray(estimates, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("at least two one-dimensional estimates are required")
        price = float(np.mean(values))
        sample_variance = float(np.var(values, ddof=1))
        estimator_variance = sample_variance / values.size
        std_error = float(np.sqrt(estimator_variance))
        ci_lower, ci_upper = self.confidence_interval(price, std_error)
        result = {
            "price": price,
            "std_error": std_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "runtime": perf_counter() - start_time,
            "variance": estimator_variance,
            "sample_variance": sample_variance,
            "n_paths": self.n_paths if n_paths is None else int(n_paths),
            "effective_samples": int(values.size),
            "method": method,
        }
        if extra:
            result.update(extra)
        return result
    

    def standard_mc(self, option_type: str = "call") -> dict:
        start = perf_counter()
        terminal = self.model.simulate(StandardNormalSampler(self.n_paths, seed=self.seed))
        return self._result(self._discounted_payoffs(terminal, option_type), start, "standard")
    

    def antithetic_mc(self, option_type: str = "call") -> dict:
        if self.n_paths % 2:
            raise ValueError("antithetic_mc requires an even n_paths")
        start = perf_counter()
        pairs = self.n_paths // 2
        base_draws = StandardNormalSampler(pairs, seed=self.seed).sample()
        original, antithetic = AntitheticVariates.transform(base_draws)
        original_payoffs = self._discounted_payoffs(self.model.get_stock_price(self.T, original), option_type)
        antithetic_payoffs = self._discounted_payoffs(self.model.get_stock_price(self.T, antithetic), option_type)
        pair_estimates = 0.5 * (original_payoffs + antithetic_payoffs)
        return self._result(
            pair_estimates,
            start,
            "antithetic",
            extra={"pairs": pairs},
        )


    def control_variate_mc(self, option_type: str = "call") -> dict:
        start = perf_counter()
        terminal = self.model.simulate(
            StandardNormalSampler(self.n_paths, seed=self.seed)
        )
        payoffs = self._discounted_payoffs(terminal, option_type)
        discounted_stock = np.exp(-self.r * self.T) * terminal
        control = ControlVariates(self.S0, self.K, self.r, self.sigma, self.T)
        beta = control.optimal_coefficient(payoffs, discounted_stock)
        adjusted = control.adjusted_estimates(payoffs, discounted_stock, beta)
        return self._result(
            adjusted,
            start,
            "control_variate",
            extra={"beta": beta},
        )

    def stratified_mc(
        self, option_type: str = "call", n_strata: int = 10
    ) -> dict:
        if self.n_paths % n_strata:
            raise ValueError("n_paths must be divisible by n_strata")
        start = perf_counter()
        samples_per_stratum = self.n_paths // n_strata
        stratifier = StratifiedSampling(
            n_strata, samples_per_stratum, seed=self.seed
        )
        uniform = stratifier.stratified_uniform()
        epsilon = np.finfo(float).eps
        draws = norm.ppf(np.clip(uniform, epsilon, 1.0 - epsilon))
        terminal = self.model.get_stock_price(self.T, draws)
        payoffs = self._discounted_payoffs(terminal, option_type)
        by_stratum = payoffs.reshape(n_strata, samples_per_stratum)
        stratum_means = np.mean(by_stratum, axis=1)
        price = float(np.mean(stratum_means))
        if samples_per_stratum > 1:
            stratum_variances = np.var(by_stratum, axis=1, ddof=1)
            estimator_variance = float(
                np.sum(stratum_variances / samples_per_stratum) / n_strata**2
            )
        else:
            estimator_variance = float(np.var(stratum_means, ddof=1) / n_strata)
        std_error = float(np.sqrt(estimator_variance))
        ci_lower, ci_upper = self.confidence_interval(price, std_error)
        return {
            "price": price,
            "std_error": std_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "runtime": perf_counter() - start,
            "variance": estimator_variance,
            "sample_variance": float(np.var(payoffs, ddof=1)),
            "n_paths": self.n_paths,
            "effective_samples": n_strata,
            "method": "stratified",
            "n_strata": n_strata,
        }

    def quasi_mc(
        self,
        option_type: str = "call",
        method: str = "sobol",
        max_replications: int = 8,
    ) -> dict:
        if method not in {"sobol", "halton"}:
            raise ValueError("method must be 'sobol' or 'halton'")
        if max_replications < 2:
            raise ValueError("max_replications must be at least 2")
        start = perf_counter()
        replications = min(int(max_replications), self.n_paths)
        base_count, remainder = divmod(self.n_paths, replications)
        replication_path_counts = [
            base_count + (index < remainder) for index in range(replications)
        ]
        sampler_class = SobolSampler if method == "sobol" else HaltonSampler
        replication_estimates = []
        replicate_seeds = (
            [None] * replications
            if self.seed is None
            else [
                int(child.generate_state(1, dtype=np.uint32)[0])
                for child in np.random.SeedSequence(self.seed).spawn(replications)
            ]
        )
        for index, path_count in enumerate(replication_path_counts):
            replicate_seed = replicate_seeds[index]
            sampler = sampler_class(path_count, seed=replicate_seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                terminal = self.model.simulate(sampler)
            payoff = self._discounted_payoffs(terminal, option_type)
            replication_estimates.append(float(np.mean(payoff)))
        return self._result(
            np.asarray(replication_estimates),
            start,
            f"quasi_{method}",
            extra={
                "qmc_method": method,
                "replications": replications,
                "replication_path_counts": tuple(replication_path_counts),
            },
        )
