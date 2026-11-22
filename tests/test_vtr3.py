"""Smoke-test the pinned upstream solvers in the MuJoCo benchmark."""

import os
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from vtr3_mpc import SCRIPTS, load_solvers


class TestUpstreamMPC(unittest.TestCase):
    def test_modified_solver_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            source = checkout / SCRIPTS / 'bicycle_solver_pt.py'
            source.parent.mkdir(parents=True)
            source.write_text('raise AssertionError("unverified source ran")')
            with self.assertRaisesRegex(ValueError, 'differs from VT&R3 revision'):
                load_solvers(checkout)

    def test_upstream_solves_in_mujoco(self):
        checkout = Path(os.environ.get('VTR3_CHECKOUT', ROOT / 'runs' / 'vtr3'))
        if not checkout.exists() and 'VTR3_CHECKOUT' not in os.environ:
            self.skipTest('Pinned VT&R3 checkout is required for the integration test')
        from benchmark import run_episode
        from config import Scenario

        scenario = Scenario('flat')
        result = run_episode(replace(scenario, distance=3, speed=0.75), 0, None, load_solvers(checkout))
        self.assertEqual(result['success'], 1, result)
        self.assertEqual(result['solver_failures'], 0, result)
        self.assertLess(result['path_peak_cm'], 1)
