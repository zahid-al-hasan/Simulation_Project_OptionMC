"""Tests for variance_reduction.py — antithetic, control variate, stratified, QMC."""

import numpy as np
import pytest
from optionmc.variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    StratifiedSampling,
    QuasiMonteCarlo,
)


class TestAntitheticVariates:

    def test_output_count(self):
        """Should double the effective sample size."""
        # TODO
        pass

    def test_negation_property(self):
        """Z_antithetic should equal -Z_original."""
        # TODO
        pass

    def test_reduces_variance(self):
        """Variance of antithetic estimator should be lower than standard."""
        # TODO
        pass


class TestControlVariates:

    def test_optimal_beta(self):
        """Beta should minimize variance of adjusted estimator."""
        # TODO
        pass

    def test_adjusted_variance_lower(self):
        # TODO
        pass


class TestStratifiedSampling:

    def test_coverage(self):
        """Each stratum should have samples."""
        # TODO
        pass

    def test_output_count(self):
        # TODO
        pass


class TestQuasiMonteCarlo:

    def test_sobol_qmc(self):
        # TODO
        pass

    def test_halton_qmc(self):
        # TODO
        pass
