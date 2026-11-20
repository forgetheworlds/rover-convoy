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




HELD_OUT = (
    Scenario('flat'),
    Scenario('s-bend', 1.4, 0.31),
    Scenario('475kg-crest', 0.65, 0.28, slope_deg=7.8),
    Scenario('low-grip', 0.9, 0.33, friction=0.39),
    Scenario('heavy-offset-estimates', 1.15, 0.29, 605, 0.74, 3.5, 0.025, 2),
    Scenario('combined', 1.45, 0.30, 565, 0.46, 7.8, 0.028, 2),
    Scenario('figure-eight-two-laps', 0, 0.26, 475, 0.64, 7.8, 0.022, 1, True),
)

SPEED_SCENARIOS = (
    Scenario('flat', distance=60.0),
    Scenario('bends', amplitude=1.45, frequency=0.27, distance=60.0),
    Scenario('475kg-crest', amplitude=-0.65, frequency=0.24, slope_deg=7.8, distance=60.0, crest_x=34.0),
    Scenario('loose-crest', amplitude=1.05, frequency=0.26, mass=540, friction=0.39,
             slope_deg=8.4, pose_noise=0.022, state_delay=1, distance=60.0, crest_x=38.0),
)

# Speed-conditioned training emphasizes the high-speed cornering problem.
# Speed selection is a curriculum, NOT a speed constraint at control time.
ADAPTIVE_SPEEDS = (0.75, 1.5, 2.5, 3.0, 3.0, 3.0)
