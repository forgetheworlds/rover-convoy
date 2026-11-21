"""A shared route supervisor for a coupled stop, role swap, and reverse approach."""

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch

from benchmark import DistributedMPC
from config import DT, SPAN, SPEED, Scenario
from field_pair import HIDDEN_INPUTS, OBS_SIZE, FieldPairEnv
from routes import Route, angle_wrap, projected_distance, route_poses
from vtr3_mpc import load_solvers

STOP = np.array([-1., 0., -1., 0.])
DOOR_LOCAL = np.array([-4.8125, 0., .82])


def reverse_branch(forward):
    """Return through the taught junction, then follow a short docking spur."""
    junction = 8.0
    distance = np.linspace(forward.finish + SPAN + 3., junction, 400)
    poses = route_poses(forward, distance)
    xy = np.column_stack([SPAN - poses[:, 0], -poses[:, 1]])
    heading = poses[-1, 2]
    tangent = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-tangent[1], tangent[0]])
    along = np.linspace(0., 21., 421)[1:]
    shift = .7 * (1 - np.cos(np.pi * np.minimum(along / 6., 1.))) / 2
    spur = xy[-1] + along[:, None] * tangent + shift[:, None] * normal
    approach_length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    route = Route.from_points(np.vstack([xy, spur]), approach_length + 21.)
    target_station = float(projected_distance(route, (xy[-1] + 13 * tangent + .7 * normal)[None, :])[0])
    return route, target_station


def switch_roles(pair, route):
    """Change the route and role ordering while preserving the physical state."""
    env = pair.env
    env.reverse = True
    env.route = route
    physical = env.data.qpos[env.free_qpos[::-1, None] + np.arange(2)]
    xy = np.column_stack([SPAN - physical[:, 0], -physical[:, 1]])
    stations = projected_distance(route, xy)
    env.indices = np.searchsorted(route.s, stations).clip(0, len(route.s) - 1)
    env.observed_indices = env.indices.copy()
    env.measurements.clear()
    env.pose_history.clear()
    env.pose_bias.fill(0.)
    env.ticks = 0
    env.max_steps = int((route.finish / SPEED + 30) / DT)
    observation, env.metrics = env._observe()
    pair.leader_index = int(env.indices[0])
    return pair._observe(observation, STOP[:2])


def cargo_door(env):
    body = env.data.body('cargo')
    rotation = body.xmat.reshape(3, 3)
    door = body.xpos + rotation @ DOOR_LOCAL
    return np.array([SPAN - door[0], -door[1], np.arctan2(rotation[1, 0], rotation[0, 0])])


def run_maneuver(scenario, seed, policy, solvers, *, observation='full'):
    pair = FieldPairEnv(scenario, observation=observation if policy is not None else 'full',
                        direction_input=policy is not None and policy.observation_space.shape[0] > OBS_SIZE)
    env = pair.env
    observation, _ = pair.reset(seed=seed)
    forward = DistributedMPC(scenario.speed, solvers) if solvers else None
    reverse = DistributedMPC(scenario.speed, solvers, reverse=True) if solvers else None
    slides = []
    steps = 0

    def update(action):
        nonlocal observation, steps
        observation, _, ended, timeout, info = pair.step(action)
        steps += 1
        slides.append(info['slider'])
        return ended or timeout, info

    def failed(reason):
        return dict(success=False, stage=reason, time_s=round(steps * DT, 2),
                    slide_peak_cm=round(100 * max(map(abs, slides)), 3))

    def stop():
        for _ in range(65):
            _, info = update(STOP)
            if info['constraint_violation']:
                return info['stop_reason']
            if np.max(np.abs(info['body_speed'])) < .022 and max(abs(env.commands[::2])) < .005:
                return ''
        return 'could-not-stop'

    try:
        route, target = reverse_branch(env.route)
        while True:
            action = forward(env, SPEED) if forward else policy.predict(observation, deterministic=True)[0]
            done, info = update(action)
            if done:
                if not info['success']:
                    return failed('forward/' + info['stop_reason'])
                break
        reason = stop()
        if reason:
            return failed('switch/' + reason)
        observation = switch_roles(pair, route)
        while True:
            estimate = env.estimated_states[1]
            station = projected_distance(route, estimate[None, :2])[0]
            braking_distance = max(0., estimate[3])**2 / .6
            if station + braking_distance >= target:
                break
            action = reverse(env, SPEED) if reverse else policy.predict(observation, deterministic=True)[0]
            done, info = update(action)
            if done:
                return failed('reverse/' + (info['stop_reason'] or 'missed-stop-point'))
        reason = stop()
        if reason:
            return failed('dock-stop/' + reason)
        actual = cargo_door(env)
        goal = route_poses(route, np.array([target + 4.8125]))[0]
        error = actual[:2] - goal[:2]
        tangent = np.array([np.cos(goal[2]), np.sin(goal[2])])
        along = float(error @ tangent)
        across = float(error @ np.array([-tangent[1], tangent[0]]))
        yaw = float(np.rad2deg(angle_wrap(actual[2] - goal[2])))
        success = abs(along) <= .25 and abs(across) <= .15 and abs(yaw) <= 5
        return dict(success=success, stage='dock-aligned' if success else 'dock-tolerance',
                    time_s=round(steps * DT, 2), slide_peak_cm=round(100 * max(map(abs, slides)), 3),
                    along_cm=round(100 * along, 2), lateral_cm=round(100 * across, 2),
                    yaw_deg=round(yaw, 2))
    finally:
        pair.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--vtr3', type=Path, required=True)
    parser.add_argument('--observation', choices=sorted(HIDDEN_INPUTS), default='pose-only')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = PPO.load(args.model, device='cpu')
    solvers = load_solvers(args.vtr3)
    cases = (
        Scenario('curved-approach', amplitude=.54, frequency=.21, distance=24, speed=.75),
        Scenario('graded-approach', amplitude=-.63, frequency=.23, distance=26, speed=.75,
                 slope_deg=4.2, crest_x=17., friction=.67, pose_noise=.015, state_delay=1),
    )
    rows = []
    for scenario in cases:
        for name, model, reference in [('learned', policy, None), ('upstream-vtr3', None, solvers)]:
            row = dict(controller=name, scenario=scenario.name,
                       **run_maneuver(scenario, 1221, model, reference, observation=args.observation))
            rows.append(row)
            print(row, flush=True)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with args.out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
