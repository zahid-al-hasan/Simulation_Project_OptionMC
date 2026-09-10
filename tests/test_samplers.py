"""Tests for samplers.py — random and quasi-random number generators."""

import numpy as np
import pytest
from optionmc.samplers import StandardNormalSampler, SobolSampler, HaltonSampler


class TestStandardNormalSampler:

    def test_shape(self):
        # TODO
        pass

    def test_mean_close_to_zero(self):
        # TODO
        pass

    def test_std_close_to_one(self):
        # TODO
        pass


class TestSobolSampler:

    def test_output_shape(self):
        # TODO
        pass

    def test_values_in_unit_cube(self):
        """Sobol points before inverse-CDF mapping should be in [0,1]."""
        # TODO
        pass


class TestHaltonSampler:

    def test_output_shape(self):
        # TODO
        pass

    def test_low_discrepancy(self):
        """Halton sequence should have lower discrepancy than random."""
        # TODO
        pass
