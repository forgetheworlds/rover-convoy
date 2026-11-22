"""Load the original VT&R3 solver sources and supply MuJoCo benchmark inputs."""

from hashlib import sha256
from pathlib import Path
import runpy

import numpy as np

from config import CORRIDOR, DT, MAX_STEER, SLEW, SPAN, WHEELBASE

REVISION = '76433e3e1a275511ece5097d832534f6a9202da1'
SCRIPTS = Path('main/src/vtr_path_planning/scripts')
STAGES, STEP = 15, 0.25
POSE_VALUES = 3 * (STAGES + 1)

_SOURCE = (
    ('bicycle_solver_pt.py', 'f03fbae93c6f71310bfcd0433996b7aa5d4f5051db1fb266a57f617af7315065'),
    ('bicycle_follower_solver_pt.py', 'e13eb958afa006b4e29b7f65584c46600a4813caa1192a18c73721507c75b33d'),
)
# Forward weights from config/lelr_convoy.yaml. Gap weight and tolerance use
# the pinned follower header defaults.
WEIGHTS = (WHEELBASE, 15.0, 4.0, 5.0, 0.1, 1.0, 4.5, 7.0, 1.0)
GAP_WEIGHT, GAP_MARGIN = 1.0, 1.0


def load_solvers(checkout):
    """Check both source hashes before executing either upstream file."""
    sources = []
    for filename, expected in _SOURCE:
        path = checkout / SCRIPTS / filename
        if not path.is_file():
            raise ValueError(f'Missing {path}. See README for the pinned VT&R3 checkout.')
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{path} differs from VT&R3 revision {REVISION}.')
        sources.append(path)
    return tuple(runpy.run_path(str(path))['solver'] for path in sources)


class MotionMPC:
    def __init__(self, solver, speed_limit, follower=False):
        self.solver = solver
        self.follower = follower
        self.guess = None
        self.lower_x = np.r_[np.full(POSE_VALUES, -np.inf), np.tile([0, -MAX_STEER], STAGES)]
        self.upper_x = np.r_[np.full(POSE_VALUES, np.inf), np.tile([speed_limit, MAX_STEER], STAGES)]
        # The pinned solver packs dynamics, corridor errors, the first command,
        # N-2 command differences, then follower-only distance constraints.
        rates = np.tile(SLEW[:2] * STEP / DT, STAGES - 2)
        self.lower_g = np.r_[np.zeros(POSE_VALUES), np.repeat(-CORRIDOR, STAGES), [0, 0], -rates]
        self.upper_g = np.r_[np.zeros(POSE_VALUES), np.repeat(CORRIDOR, STAGES), [0, 0], rates]
        if follower:
            self.lower_g = np.r_[self.lower_g, np.repeat(SPAN - GAP_MARGIN, STAGES)]
            self.upper_g = np.r_[self.upper_g, np.repeat(SPAN + GAP_MARGIN, STAGES)]
        self.first = slice(POSE_VALUES + STAGES, POSE_VALUES + STAGES + 2)
        p_size = 3 + 3 * STAGES + 2 + len(WEIGHTS) + (3 * STAGES + 2 if follower else STAGES)
        if (solver.size1_in('p'), solver.size1_in('lbg'), solver.size1_in('x0')) != (
            p_size, len(self.lower_g), len(self.lower_x)
        ):
            raise ValueError('Unexpected VT&R3 solver interface.')

    def solve(self, initial, previous, refs, lead_plan=None):
        pose = initial[:3]
        base = np.r_[pose, refs.ravel(), previous]
        if self.follower:
            parameters = np.r_[base, lead_plan.ravel(), SPAN, WEIGHTS, GAP_WEIGHT]
        else:
            parameters = np.r_[base, np.ones(STAGES), WEIGHTS]
        self.lower_g[self.first] = previous - SLEW[:2]
        self.upper_g[self.first] = previous + SLEW[:2]
        guess = self.guess if self.guess is not None else np.r_[pose, refs.ravel(), np.tile(previous, STAGES)]
        try:
            solution = self.solver(x0=guess, p=parameters, lbx=self.lower_x, ubx=self.upper_x,
                                   lbg=self.lower_g, ubg=self.upper_g)
            if self.solver.stats()['success']:
                values = np.asarray(solution['x']).ravel()
                if np.isfinite(values).all():
                    self.guess = values
                    return values[POSE_VALUES:POSE_VALUES + 2], values[3:POSE_VALUES].reshape(STAGES, 3), True
        except RuntimeError:
            pass
        self.guess = None
        # Decelerate through the common actuator interface after an infeasible solve.
        return np.array([0.0, previous[1]]), np.tile(pose, (STAGES, 1)), False
