"""Check the physical sign transform and the coordinated role swap."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from config import CONDITIONED_CRUISE, Scenario
from field_pair import FieldPairEnv, OBS_SIZE
from maneuver import reverse_branch, switch_roles


class TestReverse(unittest.TestCase):
    def test_backing_uses_negative_wheel_commands_in_swapped_order(self):
        pair = FieldPairEnv(Scenario('flat'), reverse=True, direction_input=True)
        try:
            obs, _ = pair.reset(seed=2)
            self.assertEqual(obs.shape, (OBS_SIZE + 1,))
            self.assertEqual(obs[-1], -1)
            action = np.array([.60 / .75 - CONDITIONED_CRUISE, 0.,
                               .35 / .75 - CONDITIONED_CRUISE, 0.])
            for _ in range(18):
                pair.step(action)
            physical_a = pair.env.data.ctrl[:6]
            physical_b = pair.env.data.ctrl[8:14]
            self.assertTrue(np.all(physical_a < 0) and np.all(physical_b < 0))
            self.assertLess(np.mean(physical_b), np.mean(physical_a))
        finally:
            pair.close()

    def test_role_swap_preserves_positions_and_velocities(self):
        pair = FieldPairEnv(Scenario('flat', distance=24.), direction_input=True)
        try:
            pair.reset(seed=1)
            state = pair.env.data.qpos.copy()
            velocity = pair.env.data.qvel.copy()
            route, _ = reverse_branch(pair.env.route)
            obs = switch_roles(pair, route)
            np.testing.assert_array_equal(pair.env.data.qpos, state)
            np.testing.assert_array_equal(pair.env.data.qvel, velocity)
            self.assertTrue(pair.env.reverse)
            self.assertEqual(obs[-1], -1)
        finally:
            pair.close()
