"""Estimated leader rollout and Euclidean follower references."""


import numpy as np

from config import CONDITIONED_CRUISE, DT, MAX_STEER, SLEW, WHEELBASE
from routes import angle_wrap, follower_references
from vtr3_mpc import STAGES, STEP

REFERENCE_STAGES = [2, 6, 14]  # 0.75, 1.75, and 3.75 seconds
PREVIEW_FEATURES = 3 * (STAGES + len(REFERENCE_STAGES))


def _advance(pose, speed, steer, duration):
    turn = duration * speed * np.tan(steer) / WHEELBASE
    direction = pose[2] + turn / 2
    delta = duration * speed * np.array([np.cos(direction), np.sin(direction)])
    return np.r_[pose[:2] + delta, pose[2] + turn]


def forecast_leader(estimated_state, action, speed_limit, estimate_age):
    """Predict 15 poses under the current learned command and actuator slew."""
    x, y, heading, speed, steer = estimated_state
    pose = _advance(np.array([x, y, heading]), speed, steer, estimate_age)
    target = np.array([
        speed_limit * np.clip(CONDITIONED_CRUISE + action[0], 0, 1),
        MAX_STEER * np.clip(action[1], -1, 1),
    ])
    previous = np.array([speed, steer])
    rates = SLEW[:2] * STEP / DT
    poses = np.empty((STAGES, 3))
    for index in range(STAGES):
        updated = previous + np.clip(target - previous, -rates, rates)
        mid_speed, mid_steer = (updated + previous) / 2
        pose = _advance(pose, mid_speed, mid_steer, STEP)
        poses[index] = pose
        previous = updated
    return poses


def _local_poses(poses, follower_pose):
    dx, dy = (poses[:, :2] - follower_pose[:2]).T
    angle = follower_pose[2]
    return np.column_stack([
        (np.cos(angle) * dx + np.sin(angle) * dy) / 10,
        (-np.sin(angle) * dx + np.cos(angle) * dy) / 5,
        angle_wrap(poses[:, 2] - angle),
    ]).ravel()


def rollout_features(env, leader_action, leader_index):
    """Encode the latest published leader rollout and the matching route references."""
    leader, follower = env.estimated_states
    age = env.estimate_age
    plan = forecast_leader(leader, leader_action, env.speed_limit, age)
    follower_pose = _advance(follower[:3], follower[3], follower[4], age)
    route = env.route
    hints = None
    if route.figure8:
        leader_pose = _advance(leader[:3], leader[3], leader[4], age)
        leader_index, start, _ = route.track(leader_pose[:2], leader_index)
        xy = np.vstack((leader_pose[:2], plan[:, :2]))
        hints = start + np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    references = follower_references(route, plan[:, :2], lead_distance_hint=hints)
    return (np.r_[_local_poses(plan, follower_pose),
                  _local_poses(references[REFERENCE_STAGES], follower_pose)], leader_index)
