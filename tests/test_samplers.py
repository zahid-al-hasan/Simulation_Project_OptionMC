"""Tests for pseudorandom and low-discrepancy samplers."""

import numpy as np
import pytest
from scipy.stats import qmc

from optionmc.samplers import HaltonSampler, SobolSampler, StandardNormalSampler


class TestStandardNormalSampler:
    def test_shape_and_reproducibility(self):
        first = StandardNormalSampler(1_000, seed=42).sample()
        second = StandardNormalSampler(1_000, seed=42).sample()
        assert first.shape == (1_000,)
        np.testing.assert_array_equal(first, second)

    def test_moments(self):
        draws = StandardNormalSampler(100_000, seed=7).sample()
        assert abs(np.mean(draws)) < 0.01
        assert abs(np.std(draws, ddof=1) - 1.0) < 0.01

    @pytest.mark.parametrize("n_paths", [0, -1, 1.5])
    def test_invalid_path_count(self, n_paths):
        with pytest.raises(ValueError):
            StandardNormalSampler(n_paths)


class TestSobolSampler:
    def test_normal_output_shape_and_finiteness(self):
        draws = SobolSampler(256, seed=42).sample()
        assert draws.shape == (256,)
        assert np.all(np.isfinite(draws))

    def test_uniform_points_are_in_unit_cube(self):
        points = SobolSampler(256, seed=42, d=2).uniform_sample()
        assert points.shape == (256, 2)
        assert np.all((points >= 0.0) & (points <= 1.0))


class TestHaltonSampler:
    def test_multidimensional_shape(self):
        draws = HaltonSampler(128, seed=42, d=3).sample()
        assert draws.shape == (128, 3)
        assert np.all(np.isfinite(draws))

    def test_low_discrepancy(self):
        halton = HaltonSampler(256, seed=42, d=2).uniform_sample()
        random = np.random.default_rng(42).random((256, 2))
        assert qmc.discrepancy(halton) < qmc.discrepancy(random)
