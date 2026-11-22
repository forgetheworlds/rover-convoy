"""The speed sweep changes the command interval, not the physical limits."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from config import Scenario
from field_pair import FieldPairEnv


class TestControlFrequency(unittest.TestCase):
    def test_same_slew_and_pose_delay_at_all_rates(self):
        for hz in (10, 20, 50):
            with self.subTest(hz=hz):
                env = FieldPairEnv(Scenario(state_delay=1), control_hz=hz)
                try:
                    env.reset(seed=2)
                    t0 = env.env.data.time
                    for _ in range(hz):
                        env.step(np.array([1., 0., 1., 0.]))
                    np.testing.assert_allclose(env.env.commands[::2], [.30, .30], atol=1e-10)
                    self.assertAlmostEqual(env.env.estimate_age, .10)
                    self.assertAlmostEqual(env.env.data.time - t0, 1.)
                finally:
                    env.close()
