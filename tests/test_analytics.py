"""Tests for analytics.py — convergence analysis and metrics."""

import numpy as np
import pytest
from optionmc.analytics import (
    compute_relative_error,
    variance_reduction_ratio,
    efficiency_ratio,
)


class TestRelativeError:

    def test_zero_error(self):
        assert compute_relative_error(10.0, 10.0) == 0.0
        # TODO
        pass

    def test_known_error(self):
        # TODO
        pass


class TestVarianceReductionRatio:

    def test_vrr_gt_one(self):
        """VRR should be > 1 when variance reduction works."""
        # TODO
        pass


class TestEfficiencyRatio:

    def test_efficiency_gt_one(self):
        # TODO
        pass
