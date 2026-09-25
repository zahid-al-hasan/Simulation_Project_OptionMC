"""Monte Carlo pricing engine for European options."""

from time import perf_counter

import numpy as np
from scipy.stats import norm, t as student_t

from optionmc.models import GeometricBrownianMotion
from optionmc.samplers import HaltonSampler, SobolSampler, StandardNormalSampler
from optionmc.variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    StratifiedSampling,
)


SUPPORTED_METHODS = (
    "standard", "antithetic", "control_variate", "stratified",
    "sobol", "halton", "antithetic_control",
)
SUPPORTED_OPTION_TYPES = ("call", "put")


def price_once(pricer, method: str, option_type: str = "call") -> dict:
    """One dispatch point for demos and repeated experiments."""
    if method in {"sobol", "halton"}:
        return pricer.quasi_mc(option_type, method=method)
    methods = {
        "standard": pricer.standard_mc,
        "antithetic": pricer.antithetic_mc,
        "control_variate": pricer.control_variate_mc,
        "stratified": pricer.stratified_mc,
        "antithetic_control": pricer.antithetic_control_mc,
    }
    if method not in methods:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    return methods[method](option_type)


class OptionPricing:
    """Price European calls and puts with Monte Carlo methods."""

    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        n_paths: int,
        seed: int | None = None,
    ):
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

    def _discounted_payoffs(
        self, terminal_prices: np.ndarray, option_type: str
    ) -> np.ndarray:
        normalized = self._validate_option_type(option_type)
        prices = np.asarray(terminal_prices, dtype=float)
        if normalized == "call":
            payoff = np.maximum(prices - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - prices, 0.0)
        return np.exp(-self.r * self.T) * payoff

    def confidence_interval(
        self, mean: float, std_error: float, confidence: float = 0.95
    ) -> tuple[float, float]:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if std_error < 0:
            raise ValueError("std_error cannot be negative")
        critical_value = norm.ppf(0.5 + confidence / 2.0)
        margin = critical_value * std_error
        return float(mean - margin), float(mean + margin)

    def _result(
        self,
        estimates: np.ndarray,
        start_time: float,
        method: str,
        n_paths: int | None = None,
        extra: dict | None = None,
    ) -> dict:
        values = np.asarray(estimates, dtype=float)
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
            "payoff_evaluations": self.n_paths if n_paths is None else int(n_paths),
            "pilot_evaluations": 0,
        }
        if extra:
            result.update(extra)
        return result

    def standard_mc(self, option_type: str = "call") -> dict:
        start = perf_counter()
        terminal = self.model.simulate(
            StandardNormalSampler(self.n_paths, seed=self.seed)
        )
        return self._result(
            self._discounted_payoffs(terminal, option_type), start, "standard"
        )

    def antithetic_mc(self, option_type: str = "call") -> dict:
        if self.n_paths % 2 or self.n_paths < 4:
            raise ValueError("antithetic_mc requires an even n_paths of at least four")
        start = perf_counter()
        pairs = self.n_paths // 2
        base_draws = StandardNormalSampler(pairs, seed=self.seed).sample()
        original, antithetic = AntitheticVariates.transform(base_draws)
        original_payoffs = self._discounted_payoffs(
            self.model.get_stock_price(self.T, original), option_type
        )
        antithetic_payoffs = self._discounted_payoffs(
            self.model.get_stock_price(self.T, antithetic), option_type
        )
        pair_estimates = 0.5 * (original_payoffs + antithetic_payoffs)
        return self._result(
            pair_estimates,
            start,
            "antithetic",
            extra={"pairs": pairs},
        )

    def control_variate_mc(self, option_type: str = "call") -> dict:
        """Fit beta on an independent pilot included in the total budget."""
        return self._controlled_mc(option_type, paired=False)

    def antithetic_control_mc(self, option_type: str = "call") -> dict:
        """Apply a pilot-fitted stock control to independent antithetic pairs."""
        return self._controlled_mc(option_type, paired=True)

    def _controlled_mc(self, option_type: str, paired: bool) -> dict:
        unit_cost = 2 if paired else 1
        if self.n_paths % unit_cost or self.n_paths < 4 * unit_cost:
            raise ValueError("control pricing needs at least four units (pairs if antithetic)")
        start = perf_counter()
        units = self.n_paths // unit_cost
        pilot_units = max(2, units // 10)
        pilot_seed, production_seed = np.random.SeedSequence(self.seed).spawn(2)

        def observations(count, seed):
            draws = np.random.default_rng(seed).standard_normal(count)
            terminal = self.model.get_stock_price(self.T, draws)
            payoff = self._discounted_payoffs(terminal, option_type)
            stock = np.exp(-self.r * self.T) * terminal
            if paired:
                opposite = self.model.get_stock_price(self.T, -draws)
                payoff = (payoff + self._discounted_payoffs(opposite, option_type)) / 2
                stock = (stock + np.exp(-self.r * self.T) * opposite) / 2
            return payoff, stock

        pilot_payoffs, pilot_stock = observations(pilot_units, pilot_seed)
        control = ControlVariates(self.S0, self.K, self.r, self.sigma, self.T)
        beta = (0.0 if np.var(pilot_stock) == 0 else
                control.optimal_coefficient(pilot_payoffs, pilot_stock))
        payoffs, discounted_stock = observations(units - pilot_units, production_seed)
        adjusted = control.adjusted_estimates(payoffs, discounted_stock, beta)
        return self._result(
            adjusted,
            start,
            "antithetic_control" if paired else "control_variate",
            extra={"beta": beta, "pilot_evaluations": pilot_units * unit_cost,
                   "production_evaluations": (units - pilot_units) * unit_cost},
        )

    def stratified_mc(
        self, option_type: str = "call", n_strata: int | None = None
    ) -> dict:
        if n_strata is None:
            n_strata = min(16, self.n_paths // 2)
            while self.n_paths % n_strata:
                n_strata -= 1
        if not isinstance(n_strata, (int, np.integer)) or n_strata < 1:
            raise ValueError("n_strata must be a positive integer")
        if self.n_paths % n_strata:
            raise ValueError("n_paths must be divisible by n_strata")
        start = perf_counter()
        samples_per_stratum = self.n_paths // n_strata
        if samples_per_stratum < 2:
            raise ValueError("at least two samples per stratum are needed to estimate uncertainty")
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
        stratum_variances = np.var(by_stratum, axis=1, ddof=1)
        estimator_variance = float(
            np.sum(stratum_variances / samples_per_stratum) / n_strata**2
        )
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
            "payoff_evaluations": self.n_paths,
            "pilot_evaluations": 0,
        }

    def quasi_mc(
        self,
        option_type: str = "call",
        method: str = "sobol",
        max_replications: int = 8,
    ) -> dict:
        if method not in {"sobol", "halton"}:
            raise ValueError("method must be 'sobol' or 'halton'")
        if not isinstance(max_replications, (int, np.integer)) or max_replications < 2:
            raise ValueError("max_replications must be at least 2")
        start = perf_counter()
        replications = min(int(max_replications), self.n_paths)
        while replications > 1 and self.n_paths % replications:
            replications -= 1
        if replications < 2:
            raise ValueError("QMC budget must support at least two equal-size replications")
        replication_path_counts = [self.n_paths // replications] * replications
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
            terminal = self.model.simulate(sampler)
            payoff = self._discounted_payoffs(terminal, option_type)
            replication_estimates.append(float(np.mean(payoff)))
        result = self._result(
            np.asarray(replication_estimates),
            start,
            f"quasi_{method}",
            extra={
                "qmc_method": method,
                "replications": replications,
                "replication_path_counts": tuple(replication_path_counts),
            },
        )
        # Replicate means are the independent observations, not individual QMC points.
        margin = student_t.ppf(0.975, replications - 1) * result["std_error"]
        result.update(ci_lower=result["price"] - margin,
                      ci_upper=result["price"] + margin,
                      runtime=perf_counter() - start)
        return result
