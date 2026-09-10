"""Stock price models and Black-Scholes analytical solutions."""

import numpy as np
from scipy.stats import norm


class GeometricBrownianMotion:
    """Simulate terminal stock prices under GBM (risk-neutral measure)."""

    def __init__(self, S0, r, sigma, T):
        """
        Parameters
        ----------
        S0 : float – initial stock price
        r  : float – risk-free rate
        sigma : float – volatility
        T  : float – time to maturity (years)
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T

    def get_stock_price(self, t, z):
        return (self.S0 * np.exp((self.r - self.sigma**2/2) * self.t + np.sqrt(t) * z))
        pass

    def simulate(self, Z, n_paths):
        """Generate terminal prices from standard normal draws Z.

        Parameters
        ----------
        Z : np.ndarray of shape (n_paths,) – standard normal random draws
        n_paths : int – number of simulation paths

        Returns
        -------
        np.ndarray of terminal stock prices
        """

        # S_t = []
        # for draw in n_paths:
        #     s_t = self.get_stock_price(self.T, draw)
        #     S_t.append(s_t)

        # return np.array(S_t)
        pass


class BlackScholesAnalytical:
    """Closed-form Black-Scholes price for European call/put."""

    def __init__(self, S0, K, r, sigma, T):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

        self.brownian = GeometricBrownianMotion(self.S0, self.r, self.sigma, self.T)

    def _d1(self, t, z):
        stock_price = self.brownian.get_stock_price(t, z)
        return ((np.log(np.log(stock_price/self.K)) + (self.r + self.sigma**2/2) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t)))
        pass

    def _d2(self, t):
        return (self._d1 - self.sigma*np.sqrt(self.T - t))
        pass

    def call_price(self, t, z):
        stock_price = self.brownian.get_stock_price(t, z)
        return (stock_price * norm.cdf(self._d1) - self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(self._d2))
        pass

    def put_price(self, t, z):
        stock_price = self.brownian.get_stock_price(t, z)
        return (self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(-self._d2) - stock_price * norm.cdf(-self._d1))
        pass
