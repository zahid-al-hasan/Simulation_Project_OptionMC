"""Tests for samplers.py — random and quasi-random number generators."""

import numpy as np
import pytest
from scipy.stats import norm
from optionmc.samplers import StandardNormalSampler, SobolSampler, HaltonSampler


class TestStandardNormalSampler:

    def test_shape(self):
        n_paths = 4096
        sampler = StandardNormalSampler(n_paths, seed=42)
        assert sampler.sample().shape == (n_paths,)

    def test_mean_close_to_zero(self):
        n_paths = 200_000
        sampler = StandardNormalSampler(n_paths, seed=42)
        assert sampler.sample().mean() == pytest.approx(0.0, abs=0.01)

    def test_std_close_to_one(self):
        n_paths = 200_000
        sampler = StandardNormalSampler(n_paths, seed=42)
        assert sampler.sample().std(ddof=1) == pytest.approx(1.0, abs=0.01)

    def test_reproducible_with_seed(self):
        z1 = StandardNormalSampler(1000, seed=7).sample()
        z2 = StandardNormalSampler(1000, seed=7).sample()
        np.testing.assert_array_equal(z1, z2)

    def test_different_seeds_differ(self):
        z1 = StandardNormalSampler(1000, seed=7).sample()
        z2 = StandardNormalSampler(1000, seed=8).sample()
        assert not np.allclose(z1, z2)


class TestSobolSampler:

    def test_output_shape(self):
        n_paths = 1024
        assert SobolSampler(n_paths, seed=0, d=1).sample().shape == (n_paths,)

    def test_values_in_unit_cube(self):
        n_paths = 1024
        sampler = SobolSampler(n_paths, seed=0, d=1)
        u = sampler.sobol_engine.random(n_paths)
        assert u.shape == (n_paths, 1)
        assert u.min() >= 0.0 and u.max() <= 1.0

    def test_normal_mapping(self):
        z = SobolSampler(1024, seed=0, d=1).sample()
        u = norm.cdf(z)
        assert (u > 0).all() and (u < 1).all()
        assert np.isfinite(z).all()

    def test_reproducible_with_seed(self):
        z1 = SobolSampler(512, seed=3).sample()
        z2 = SobolSampler(512, seed=3).sample()
        np.testing.assert_array_equal(z1, z2)


class TestHaltonSampler:

    def test_output_shape(self):
        n_paths = 1024
        assert HaltonSampler(n_paths, seed=0, d=1).sample().shape == (n_paths,)

    def test_values_in_unit_cube(self):
        n_paths = 1024
        sampler = HaltonSampler(n_paths, seed=0, d=1)
        u = sampler.halton_engine.random(n_paths)
        assert u.shape == (n_paths, 1)
        assert u.min() >= 0.0 and u.max() <= 1.0

    def test_reproducible_with_seed(self):
        z1 = HaltonSampler(512, seed=3).sample()
        z2 = HaltonSampler(512, seed=3).sample()
        np.testing.assert_array_equal(z1, z2)

    def test_low_discrepancy(self):
        """Scrambled Halton should have lower star discrepancy than uniform random."""

        def star_discrepancy(samples):
            pts = np.sort(samples)
            n = len(pts)
            below = np.max(np.abs(np.arange(1, n + 1) / n - pts))
            above = np.max(np.abs(pts - np.arange(n) / n))
            return max(below, above)

        n_paths = 500
        u_halton = HaltonSampler(n_paths, seed=0, d=1).halton_engine.random(n_paths).ravel()
        u_random = np.random.default_rng(0).random(n_paths)
        assert star_discrepancy(u_halton) < star_discrepancy(u_random)