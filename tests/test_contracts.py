"""Contracts and deterministic smoke tests.

Run after installing requirements: python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from config import MAX_SPEED, SPEED_SCENARIOS, Scenario
from data import ConvoyAction, OBSERVATIONS, observation_names
from routes import Route


class TestInputContracts(unittest.TestCase):
    def test_observation_schema_lengths(self):
        self.assertEqual(len(observation_names("base")), OBSERVATIONS.base)
        self.assertEqual(len(observation_names("speed_conditioned")), OBSERVATIONS.speed_conditioned)
        self.assertEqual(len(observation_names("delivery")), OBSERVATIONS.delivery)
        self.assertEqual(len(observation_names("adaptive_speed")), OBSERVATIONS.adaptive_speed)
        self.assertEqual(observation_names("base")[18], "slider_position")

    def test_actions_round_trip(self):
        values = np.array([0.5, -0.2, -0.7, 0.1], dtype=np.float32)
        np.testing.assert_array_equal(ConvoyAction.from_array(values).as_array(), values)

    def test_flat_route_zero_curvature(self):
        route = Route(Scenario("flat"))
        self.assertLess(float(np.max(np.abs(route.curvature[80:-80]))), 1e-6)

    def test_corner_route_has_signed_curvature(self):
        route = Route(SPEED_SCENARIOS[1])
        self.assertGreater(float(np.max(np.abs(route.curvature))), 0.01)


class TestEnvironmentSmoke(unittest.TestCase):
    def test_all_checkpoint_observation_shapes_and_step(self):
        from environment import CargoConvoy

        modes = (
            (False, False, False, OBSERVATIONS.base),
            (True, False, False, OBSERVATIONS.speed_conditioned),
            (True, True, False, OBSERVATIONS.delivery),
            (True, True, True, OBSERVATIONS.adaptive_speed),
        )
        for speed_conditioned, delivery, adaptive, n_features in modes:
            with self.subTest(features=n_features):
                env = CargoConvoy(Scenario("flat", speed=MAX_SPEED),
                                  speed_conditioned=speed_conditioned,
                                  delivery=delivery, adaptive_speed=adaptive)
                try:
                    obs, _ = env.reset(seed=123)
                    self.assertEqual(obs.shape, (n_features,))
                    self.assertEqual(env.observation_space.shape, (n_features,))
                    self.assertTrue(np.isfinite(obs).all())
                    obs, reward, ended, timed_out, info = env.step(np.zeros(4))
                    self.assertEqual(obs.shape, (n_features,))
                    self.assertTrue(np.isfinite(obs).all())
                    self.assertTrue(np.isfinite(reward))
                    self.assertIn("slider", info)
                finally:
                    env.close()

    def test_seed_is_reproducible(self):
        from environment import CargoConvoy

        env = CargoConvoy(delivery=True, adaptive_speed=True)
        try:
            a, _ = env.reset(seed=1202)
            b, _ = env.reset(seed=1202)
            np.testing.assert_array_equal(a, b)
        finally:
            env.close()

    def test_adaptive_requires_delivery(self):
        from environment import CargoConvoy
        with self.assertRaises(ValueError):
            CargoConvoy(adaptive_speed=True)


if __name__ == "__main__":
    unittest.main()
