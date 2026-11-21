"""Human-readable data models and indexed policy input/output contracts.

The simulation deliberately uses flat NumPy arrays in its hot loop; dataclasses
below are inspection helpers, not per-timestep allocations.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str = 'flat'
    amplitude: float = 0.0
    frequency: float = 0.26
    mass: float = 475.0
    friction: float = 0.84
    slope_deg: float = 0.0
    pose_noise: float = 0.0
    state_delay: int = 0
    figure8: bool = False
    distance: float = 12.0
    crest_x: float = 14.0
    speed: float = 0.75  # Original field-trial maximum; default scenario speed
    span: float = 4.15   # Nominal center-to-center cargo span (m)
    loop_width: float = 22.0
    loop_height: float = 11.0

@dataclass(frozen=True)
class RoverCommand:
    """Normalized [-1, 1] speed and steering action for one rover."""
    speed: float
    steering: float


@dataclass(frozen=True)
class ConvoyAction:
    """Human-readable view of the policy's four-element action vector."""
    leader: RoverCommand
    follower: RoverCommand

    @classmethod
    def from_array(cls, values):
        if len(values) != 4:
            raise ValueError("Expected [leader_speed, leader_steer, follower_speed, follower_steer]")
        return cls(RoverCommand(float(values[0]), float(values[1])),
                   RoverCommand(float(values[2]), float(values[3])))

    def as_array(self):
        return np.asarray([self.leader.speed, self.leader.steering,
                           self.follower.speed, self.follower.steering], dtype=np.float32)


@dataclass(frozen=True)
class ObservationSchema:
    """Sizes of stable, positional inputs used by saved PPO checkpoints."""
    base: int = 28
    speed_conditioned: int = 29
    delivery: int = 35
    adaptive_speed: int = 41


OBSERVATIONS = ObservationSchema()

# Features for each rover, in exactly the order written by CargoConvoy._observe().
_ROVER = (
    "cross_track", "heading_error", "forward_velocity", "lateral_velocity",
    "yaw_rate", "steering", "near_route_offset", "mid_route_offset",
    "far_route_offset",
)
_SHARED = (
    "slider_position", "slider_velocity", "relative_grade", "leader_roll",
    "relative_heading_sin", "previous_leader_speed", "previous_leader_steer",
    "previous_follower_speed", "previous_follower_steer", "observation_delay",
)
_DELIVERY = ("near_future_heading", "far_future_heading", "pitch")
_CURVATURE = ("curvature_near", "curvature_mid", "curvature_far")


def observation_names(kind="adaptive_speed"):
    """Names in model-input order; useful for inspection and test assertions."""
    names = tuple("leader_" + x for x in _ROVER) + tuple("follower_" + x for x in _ROVER) + _SHARED
    if kind == "base":
        return names
    names += ("command_speed_limit",)
    if kind == "speed_conditioned":
        return names
    names += tuple(f"{rover}_{feature}" for rover in ("leader", "follower") for feature in _DELIVERY)
    if kind == "delivery":
        return names
    if kind == "adaptive_speed":
        return names + tuple(f"{rover}_{feature}" for rover in ("leader", "follower") for feature in _CURVATURE)
    raise ValueError(f"Unknown observation kind: {kind}")


def describe_observation(values):
    """Turn an observation into named values for debugging (not the hot loop)."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional policy observation")
    layouts = {
        OBSERVATIONS.base: "base",
        OBSERVATIONS.speed_conditioned: "speed_conditioned",
        OBSERVATIONS.delivery: "delivery",
        OBSERVATIONS.adaptive_speed: "adaptive_speed",
    }
    if len(values) not in layouts:
        raise ValueError(f"Unknown observation length: {len(values)}")
    return dict(zip(observation_names(layouts[len(values)]),
                    (float(value) for value in values)))
