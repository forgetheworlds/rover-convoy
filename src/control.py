"""Simple reference controller and action utilities.

The geometric controller is used for PPO warm starts; it is not a speed limiter.
"""

import numpy as np
from config import SPEED, CORRIDOR, SLIDER_LIMIT, WHEELBASE, MAX_STEER, CONDITIONED_CRUISE

def geometric_action(observation, cruise_speed=SPEED, *, speed_limit=None):
    # Same state and motor mapping. Fixed feedback checks that the task is drivable.
    slide = observation[18] * SLIDER_LIMIT
    steering = np.zeros(2)
    for rover, first in enumerate((0, 9)):
        cte = observation[first] * CORRIDOR
        near = observation[first + 7] * 1.8
        steering[rover] = np.clip(np.arctan2(2 * WHEELBASE * near, 2.0**2) - 0.16 * cte,
                                  -MAX_STEER, MAX_STEER) / MAX_STEER
    speeds = cruise_speed + np.array([0.45, -0.85]) * slide
    speed_action = ((speeds - SPEED) / 0.50 if speed_limit is None
                    else speeds / speed_limit - CONDITIONED_CRUISE)
    return np.array([speed_action[0], steering[0], speed_action[1], steering[1]])

