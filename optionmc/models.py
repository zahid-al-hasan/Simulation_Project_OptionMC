"""Stock-price models and Black-Scholes analytical solutions."""

import numpy as np
from scipy.stats import norm

from optionmc.samplers import Sampler


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    """Return Python floats for scalar calculations and arrays otherwise."""
    return float(value) if value.ndim == 0 else value


class GeometricBrownianMotion:
    """Simulate stock prices under risk-neutral geometric Brownian motion."""

    def __init__(self, S0: float, r: float, sigma: float, T: float):
        if S0 <= 0:
            raise ValueError("S0 must be positive")
        if sigma < 0:
            raise ValueError("sigma cannot be negative")
        if T < 0:
            raise ValueError("T cannot be negative")
        self.S0 = float(S0)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)

    def get_stock_price(self, t: float, z: np.ndarray | float) -> float | np.ndarray:
        """Return the stock price at time ``t`` for standard-normal draw(s) ``z``."""
        if t < 0:
            raise ValueError("t cannot be negative")
        draws = np.asarray(z, dtype=float)
        prices = self.S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * t
            + self.sigma * np.sqrt(t) * draws
        )
        return _scalar_or_array(prices)

    def simulate(self, sampler: Sampler) -> np.ndarray:
        """Generate terminal prices using the supplied normal sampler."""
        draws = np.asarray(sampler.sample(), dtype=float)
        if draws.ndim != 1:
            raise ValueError("GBM terminal simulation requires one draw per path")
        return np.asarray(self.get_stock_price(self.T, draws))


class BlackScholesAnalytical:
    """Closed-form Black-Scholes prices for European calls and puts."""

    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float):
        if K <= 0:
            raise ValueError("K must be positive")
        self.brownian = GeometricBrownianMotion(S0, r, sigma, T)
        self.S0 = self.brownian.S0
        self.K = float(K)
        self.r = self.brownian.r
        self.sigma = self.brownian.sigma
        self.T = self.brownian.T

    def _inputs(
        self, t: float, spot: float | np.ndarray | None
    ) -> tuple[np.ndarray, float]:
        if not 0 <= t <= self.T:
            raise ValueError("t must be between 0 and T")
        stock = np.asarray(self.S0 if spot is None else spot, dtype=float)
        if np.any(stock <= 0):
            raise ValueError("spot prices must be positive")
        return stock, self.T - t

    def _d1(self, t: float = 0.0, spot: float | np.ndarray | None = None):
        stock, tau = self._inputs(t, spot)
        if tau == 0 or self.sigma == 0:
            raise ValueError("d1 is undefined at expiry or when sigma is zero")
        value = (
            np.log(stock / self.K) + (self.r + 0.5 * self.sigma**2) * tau
        ) / (self.sigma * np.sqrt(tau))
        return _scalar_or_array(value)

    def _d2(self, t: float = 0.0, spot: float | np.ndarray | None = None):
        _, tau = self._inputs(t, spot)
        value = np.asarray(self._d1(t, spot)) - self.sigma * np.sqrt(tau)
        return _scalar_or_array(value)

    def call_price(
        self, t: float = 0.0, spot: float | np.ndarray | None = None
    ) -> float | np.ndarray:
        stock, tau = self._inputs(t, spot)
        if tau == 0:
            return _scalar_or_array(np.maximum(stock - self.K, 0.0))
        if self.sigma == 0:
            return _scalar_or_array(
                np.maximum(stock - self.K * np.exp(-self.r * tau), 0.0)
            )
        d1 = np.asarray(self._d1(t, stock))
        d2 = d1 - self.sigma * np.sqrt(tau)
        price = stock * norm.cdf(d1) - self.K * np.exp(-self.r * tau) * norm.cdf(d2)
        return _scalar_or_array(price)

    def put_price(
        self, t: float = 0.0, spot: float | np.ndarray | None = None
    ) -> float | np.ndarray:
        stock, tau = self._inputs(t, spot)
        if tau == 0:
            return _scalar_or_array(np.maximum(self.K - stock, 0.0))
        if self.sigma == 0:
            return _scalar_or_array(
                np.maximum(self.K * np.exp(-self.r * tau) - stock, 0.0)
            )
        d1 = np.asarray(self._d1(t, stock))
        d2 = d1 - self.sigma * np.sqrt(tau)
        price = self.K * np.exp(-self.r * tau) * norm.cdf(-d2) - stock * norm.cdf(-d1)
        return _scalar_or_array(price)
