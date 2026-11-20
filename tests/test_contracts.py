"""Contracts and deterministic smoke tests.

Run after installing requirements: python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from config import MAX_SPEED, Scenario
from data import ConvoyAction, OBSERVATIONS, describe_observation, observation_names
from routes import Route


class TestSliderReward(unittest.TestCase):
    def test_heavier_slider_penalty_is_symmetric_and_zero_at_center(self):
        from environment import delivery_reward

        zero = np.zeros(2)
        metrics = {'progress': zero, 'lateral': zero, 'tilt': 0, 'slide_speed': 0, 'slider': 0}
        for displacement in (0, 0.05, -0.05):
            metrics['slider'] = displacement
            before = delivery_reward(metrics, zero, np.zeros(4), False, False, 3)
            after = delivery_reward(metrics, zero, np.zeros(4), False, False, 3, slider_weight=0.30)
            self.assertAlmostEqual(before - after, 0.06 * abs(displacement) / 0.05)


    def test_tail_penalty_applies_only_above_ten_centimeters(self):
        from environment import delivery_reward

        state = np.zeros(2)
        metrics = {'progress': state, 'lateral': state, 'tilt': 0, 'slide_speed': 0, 'slider': 0}
        for displacement, penalty in ((0.09, 0), (0.10, 0), (-0.15, 0.125), (0.20, 0.5)):
            metrics['slider'] = displacement
            old = delivery_reward(metrics, state, np.zeros(4), False, False, 3)
            new = delivery_reward(metrics, state, np.zeros(4), False, False, 3, slider_tail_weight=0.5)
            self.assertAlmostEqual(old - new, penalty)


class TestLearnedControllers(unittest.TestCase):
    def test_leader_does_not_read_follower_features(self):
        import torch
        from learned_control import ConvoyActors

        model = ConvoyActors()
        baseline = torch.zeros((1, 41))
        varied = torch.ones((1, 41))
        varied[:, model.leader_inputs] = 0
        torch.testing.assert_close(model.forward_actor(baseline)[:, :2],
                                   model.forward_actor(varied)[:, :2])

    def test_follower_reads_only_its_own_and_link_features(self):
        import torch
        from learned_control import ConvoyActors

        model = ConvoyActors()
        baseline = torch.zeros((1, 41))
        varied = torch.ones((1, 41))
        varied[:, model.follower_inputs] = 0
        torch.testing.assert_close(model.forward_actor(baseline)[:, 2:],
                                   model.forward_actor(varied)[:, 2:])
        self.assertNotIn(18, model.leader_inputs.tolist())
        self.assertIn(18, model.follower_inputs.tolist())

    def test_loop_projection_uses_the_arclength_hint(self):
        from routes import projected_distance, route_poses

        route = Route(Scenario(figure8=True))
        # These two arclengths reach the same point on different loops.
        lap = (route.s[-1] - 8.0) / 2
        first, later = 8.0, 8.0 + lap
        crossing = route_poses(route, np.array([first]))[:, :2]
        self.assertAlmostEqual(projected_distance(route, crossing, hint=[first])[0], first, places=2)
        self.assertGreater(projected_distance(route, crossing, hint=[later])[0], first + 100)


class TestInputContracts(unittest.TestCase):
    def test_observation_schema_lengths(self):
        self.assertEqual(len(observation_names("base")), OBSERVATIONS.base)
        self.assertEqual(len(observation_names("speed_conditioned")), OBSERVATIONS.speed_conditioned)
        self.assertEqual(len(observation_names("delivery")), OBSERVATIONS.delivery)
        self.assertEqual(len(observation_names("adaptive_speed")), OBSERVATIONS.adaptive_speed)
        self.assertEqual(observation_names("base")[18], "slider_position")

    def test_observation_debug_view(self):
        values = np.zeros(OBSERVATIONS.adaptive_speed, dtype=np.float32)
        values[-1] = 1.0
        view = describe_observation(values)
        self.assertEqual(len(view), 41)
        self.assertEqual(view["follower_curvature_far"], 1.0)
        self.assertEqual(view["leader_cross_track"], 0.0)

    def test_actions_round_trip(self):
        values = np.array([0.5, -0.2, -0.7, 0.1], dtype=np.float32)
        np.testing.assert_array_equal(ConvoyAction.from_array(values).as_array(), values)

    def test_flat_route_zero_curvature(self):
        route = Route(Scenario("flat"))
        self.assertLess(float(np.max(np.abs(route.curvature[80:-80]))), 1e-6)

    def test_corner_route_has_signed_curvature(self):
        route = Route(Scenario("bend", amplitude=1.45, frequency=.27, distance=60.))
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

    def test_extended_input_preserves_first_35_values(self):
        from environment import CargoConvoy

        scenario = Scenario("flat", speed=3.0)
        old = CargoConvoy(scenario, delivery=True)
        new = CargoConvoy(scenario, delivery=True, adaptive_speed=True)
        try:
            existing, _ = old.reset(seed=123)
            extended, _ = new.reset(seed=123)
            np.testing.assert_array_equal(existing, extended[:35])
        finally:
            old.close()
            new.close()

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


class TestEntryPoints(unittest.TestCase):
    def test_command_line_help_imports(self):
        import subprocess

        root = Path(__file__).resolve().parents[1]
        for script in ("field_pair.py", "direction_trials.py", "maneuver.py", "render_convoy.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(root / "src" / script), "--help"],
                    capture_output=True, text=True, timeout=25,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_delivery_checkpoint_contract(self):
        from stable_baselines3 import PPO

        path = Path(__file__).resolve().parents[1] / "assets" / "field-forward.zip"
        policy = PPO.load(path, device="cpu")
        from field_pair import OBS_SIZE
        self.assertEqual(policy.observation_space.shape, (OBS_SIZE,))



if __name__ == "__main__":
    unittest.main()
