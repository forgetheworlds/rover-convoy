"""Physical limits and scenario presets; no action-selection logic."""

from pathlib import Path
import numpy as np

from data import Scenario

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets" / "convoy.xml"

SPEED, MAX_SPEED = 0.50, 0.75
SWEEP_SPEEDS = (0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
TRAIN_SPEEDS = (0.50, 0.75, 1.25, 1.75, 2.50)
DELIVERY_SPEEDS = (0.50, 0.75, 1.25, 1.75, 2.50, 3.0, 3.0)
SWEEP_MAX_SPEED = max(SWEEP_SPEEDS)
CONDITIONED_CRUISE = 0.85
SPAN, WHEELBASE, TRACK, RADIUS = 4.15, 1.83, 1.65, 0.3815
DT, SUBSTEPS, CORRIDOR = 0.10, 10, 0.80
MAX_STEER, SLIDER_LIMIT = 0.454, 0.50
FAILURE_COST = 16.0
DELIVERY_FAILURE_COST = 40.0
# Bounds from the public LELR control configuration.
SLEW = np.array([0.30, 0.149, 0.30, 0.149]) * DT
WHEEL_X = np.repeat([WHEELBASE, 0.61, -0.61], 2)
WHEEL_Y = np.tile([TRACK / 2, -TRACK / 2], 3)




# Speed-conditioned training emphasizes the high-speed cornering problem.
# Speed selection is a curriculum, NOT a speed constraint at control time.
ADAPTIVE_SPEEDS = (0.75, 1.5, 2.5, 3.0, 3.0, 3.0)

FIELD = (
    Scenario('field-envelope-crest', amplitude=-.61, frequency=.234, distance=60, mass=475,
             friction=.72, slope_deg=7.8, crest_x=34, speed=.75, pose_noise=.022, state_delay=1),
    Scenario('field-envelope-figure-eight', figure8=True, mass=475, friction=.74, slope_deg=7.8,
             crest_x=14, speed=.75, pose_noise=.016, state_delay=1),
)
