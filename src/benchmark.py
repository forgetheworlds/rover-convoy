"""Compare the learned joint controller with the upstream VT&R3 D-MPC solvers."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from stable_baselines3 import PPO
import torch

from config import (
    CONDITIONED_CRUISE, DT, MAX_STEER, ROOT, SPEED_SCENARIOS, SLEW, WHEELBASE,
)
from environment import CargoConvoy
from routes import angle_wrap, follower_references, projected_distance, route_poses

from vtr3_mpc import MotionMPC, REVISION, STAGES, STEP, load_solvers
SPEEDS = (0.75, 1.5, 3.0)
def reference_progress(initial_speed, cruise):
    """Reference distance at each stage with a slew-feasible cruise ramp."""
    times = STEP * np.arange(1, STAGES + 1)
    acceleration = SLEW[0] / DT
    ramp = np.minimum(times, abs(cruise - initial_speed) / acceleration)
    distance = initial_speed * ramp + np.sign(cruise - initial_speed) * acceleration * ramp**2 / 2
    return distance + cruise * (times - ramp)


class DistributedMPC:
    def __init__(self, speed_limit, solvers):
        self.speed_limit = speed_limit
        self.lead = MotionMPC(solvers[0], speed_limit)
        self.follow = MotionMPC(solvers[1], speed_limit, follower=True)
        self.failures = 0

    def __call__(self, env, cruise):
        estimated = env.estimated_states.copy()
        yaw, speed, steer = estimated[:, 2], estimated[:, 3], estimated[:, 4]
        change = env.estimate_age * speed * np.tan(steer) / WHEELBASE
        estimated[:, 0] += env.estimate_age * speed * np.cos(yaw + change / 2)
        estimated[:, 1] += env.estimate_age * speed * np.sin(yaw + change / 2)
        estimated[:, 2] = angle_wrap(yaw + change)
        initial = estimated[:, [0, 1, 2, 4]]
        previous = env.commands.reshape(2, 2)
        start = projected_distance(env.route, initial[0:1, :2])[0]
        distances = start + reference_progress(previous[0, 0], cruise)
        refs = route_poses(env.route, distances)

        lead_cmd, predicted, lead_ok = self.lead.solve(initial[0], previous[0], refs)
        refs_follow = follower_references(env.route, predicted[:, :2])
        follow_cmd, _, follow_ok = self.follow.solve(initial[1], previous[1], refs_follow, predicted)
        self.failures += int(not lead_ok) + int(not follow_ok)

        actions = np.r_[lead_cmd, follow_cmd]
        actions[::2] = actions[::2] / self.speed_limit - CONDITIONED_CRUISE
        actions[1::2] /= MAX_STEER
        return actions


def run_episode(scenario, seed, policy, solvers):
    env = CargoConvoy(scenario, delivery=True,
                      adaptive_speed=policy is not None and policy.observation_space.shape == (41,))
    observation, _ = env.reset(seed=1000 + seed)
    mpc = DistributedMPC(scenario.speed, solvers) if policy is None else None
    slide, cross, elapsed = [], [], []
    try:
        while True:
            start = perf_counter_ns()
            action = (mpc(env, scenario.speed) if mpc else policy.predict(observation, deterministic=True)[0])
            elapsed.append((perf_counter_ns() - start) * 1e-6)
            observation, _, ended, timed_out, info = env.step(action)
            slide.append(info['slider'])
            cross.append(info['lateral'])
            if ended or timed_out:
                break
    finally:
        env.close()
    lateral_rms = np.sqrt(np.mean(np.square(cross), axis=0))
    return dict(
        course=scenario.name, speed=scenario.speed, seed=seed,
        success=int(info['success']), stop=info['stop_reason'],
        progress_m=round(float(info['progress'][1]), 3), time_s=round(len(slide) * DT, 2),
        slide_rmse_cm=round(100 * np.sqrt(np.mean(np.square(slide))), 3),
        slide_peak_cm=round(100 * np.max(np.abs(slide)), 3),
        rover_a_rmse_cm=round(100 * lateral_rms[0], 3), rover_b_rmse_cm=round(100 * lateral_rms[1], 3),
        path_peak_cm=round(100 * np.max(np.abs(cross)), 3),
        controller_p95_ms=round(float(np.quantile(elapsed, 0.95)), 4),
        solver_failures=mpc.failures if mpc else 0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed', action='append', type=float, choices=SPEEDS, help='repeat to choose speeds')
    parser.add_argument('--model', action='append', type=Path, help='repeat to compare additional PPO checkpoints')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs' / 'benchmark.csv')
    parser.add_argument('--vtr3', type=Path, default=ROOT / 'runs' / 'vtr3',
                        help='checkout of the pinned VT&R3 revision (see README)')
    parser.add_argument('--course', choices=[s.name for s in SPEED_SCENARIOS])
    args = parser.parse_args()
    try:
        solvers = load_solvers(args.vtr3)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    models = args.model or [ROOT / 'assets' / 'policy.zip']
    controllers = [('VT&R3 D-MPC', None), *((p.stem, PPO.load(p, device='cpu')) for p in models)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as file:
        writer = None
        for speed in args.speed or SPEEDS:
            for scenario in SPEED_SCENARIOS:
                if args.course and scenario.name != args.course:
                    continue
                for seed in (0, 1) if scenario.name == 'loose-crest' else (0,):
                    for label, policy in controllers:
                        row = dict(controller=label, vtr3_revision=REVISION,
                                   **run_episode(replace(scenario, speed=speed), seed, policy, solvers))
                        if writer is None:
                            writer = csv.DictWriter(file, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        file.flush()
                        print(row, flush=True)


if __name__ == '__main__':
    main()
