"""Rollout interface and matched initialization checks."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from config import CONDITIONED_CRUISE
from leader_preview import forecast_leader


class TestLeaderPreview(unittest.TestCase):
    def test_forecast_has_the_same_horizon_as_upstream_mpc(self):
        action = [0.5 / 0.75 - CONDITIONED_CRUISE, 0]
        plan = forecast_leader([4.15, 0, 0, .5, 0], action, .75, 0)
        self.assertEqual(plan.shape, (15, 3))
        np.testing.assert_allclose(plan[:, 0], 4.15 + .5 * .25 * np.arange(1, 16))
        np.testing.assert_allclose(plan[:, 1:], 0)
