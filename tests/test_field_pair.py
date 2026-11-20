"""Checks for independent actors and the field-speed demonstration target."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from config import CONDITIONED_CRUISE, MAX_SPEED, SPEED
from field_pair import OBS_SIZE, FieldPairEnv, teacher_action
from learned_control import ConvoyActors
from leader_preview import PREVIEW_FEATURES


class TestFieldPair(unittest.TestCase):
    def test_rollout_is_available_to_follower_only(self):
        actors = ConvoyActors(PREVIEW_FEATURES)
        obs = torch.zeros((1, OBS_SIZE))
        changed = obs.clone()
        changed[:, 41:] = 1
        self.assertTrue(all(i >= 41 for i in actors.follower_inputs[-PREVIEW_FEATURES:]))
        self.assertTrue(all(i < 41 for i in actors.leader_inputs))
        torch.testing.assert_close(actors.forward_actor(obs)[:, :2],
                                   actors.forward_actor(changed)[:, :2])

    def test_demonstration_leader_tracks_nominal_speed(self):
        obs = np.zeros(OBS_SIZE)
        obs[18] = .2
        compressed = teacher_action(obs)
        obs[18] = -.2
        extended = teacher_action(obs)
        for action in (compressed, extended):
            self.assertAlmostEqual(MAX_SPEED * (CONDITIONED_CRUISE + action[0]), SPEED)
        self.assertLess(compressed[2], extended[2])

    def test_field_rollout_keeps_physical_command_ceiling(self):
        env = FieldPairEnv()
        try:
            obs, _ = env.reset(seed=2)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(env.env.speed_limit, MAX_SPEED)
            self.assertTrue(env.observation_space.contains(env.step(teacher_action(obs))[0]))
        finally:
            env.close()
