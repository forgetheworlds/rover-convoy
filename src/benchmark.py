"""Compare the learned joint controller with a reconstruction of the paper's D-MPC."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import casadi as ca
import numpy as np
from stable_baselines3 import PPO
import torch

from config import (
    CONDITIONED_CRUISE, CORRIDOR, DT, MAX_STEER, ROOT, SPAN, SPEED_SCENARIOS, SLEW, WHEELBASE,
)
from environment import CargoConvoy
from routes import angle_wrap, follower_references, projected_distance, route_poses

STAGES, STEP = 15, 0.25
STEER_LAG = 0.7
SPEEDS = (0.75, 1.5, 3.0)
def reference_progress(initial_speed, cruise):
    """Reference distance at each stage with a slew-feasible cruise ramp."""
    times = STEP * np.arange(1, STAGES + 1)
    acceleration = SLEW[0] / DT
    ramp = np.minimum(times, abs(cruise - initial_speed) / acceleration)
    distance = initial_speed * ramp + np.sign(cruise - initial_speed) * acceleration * ramp**2 / 2
    return distance + cruise * (times - ramp)


class MotionMPC:
    """Ackermann MPC with the public LELR weights and steering lag."""

    def __init__(self, speed_limit, follower=False):
        self.speed_limit = speed_limit
        self.follower = follower
        opt = ca.Opti()
        self.state = opt.variable(4, STAGES + 1)
        self.action = opt.variable(2, STAGES)
        self.initial = opt.parameter(4)
        self.previous = opt.parameter(2)
        self.references = opt.parameter(3, STAGES)
        self.lead_plan = opt.parameter(2, STAGES) if follower else None
        opt.subject_to(self.state[:, 0] == self.initial)
        cost = 0
        for k in range(STAGES):
            x, y, yaw, steer = ca.vertsplit(self.state[:, k])
            velocity, command = ca.vertsplit(self.action[:, k])
            next_steer = STEER_LAG * steer + (1 - STEER_LAG) * command
            turn = STEP * velocity * ca.tan(next_steer) / WHEELBASE
            # Exact constant-curvature propagation without division at zero turn.
            straight = 1 - (turn / 2)**2 / 6 + (turn / 2)**4 / 120
            next_pose = ca.vertcat(x + STEP * velocity * straight * ca.cos(yaw + turn / 2),
                                   y + STEP * velocity * straight * ca.sin(yaw + turn / 2),
                                   yaw + turn, next_steer)
            opt.subject_to(self.state[:, k + 1] == next_pose)
            opt.subject_to(opt.bounded([0, -MAX_STEER], self.action[:, k], [speed_limit, MAX_STEER]))
            change = self.action[:, k] - (self.previous if k == 0 else self.action[:, k - 1])
            rate_bound = SLEW[:2] * (1 if k == 0 else STEP / DT)
            opt.subject_to(opt.bounded(-rate_bound, change, rate_bound))

            dx, dy, dyaw = ca.vertsplit(self.state[:3, k + 1] - self.references[:, k])
            heading = self.references[2, k]
            along = ca.cos(heading) * dx + ca.sin(heading) * dy
            across = -ca.sin(heading) * dx + ca.cos(heading) * dy
            opt.subject_to(opt.bounded(-CORRIDOR, across, CORRIDOR))
            cost += 15 * across**2 + 4 * along**2 + 5 * dyaw**2 + 0.1 * velocity**2 + command**2
            cost += 4.5 * change[0]**2 + 7 * change[1]**2
            if follower:
                separation = ca.norm_2(self.state[:2, k + 1] - self.lead_plan[:, k])
                opt.subject_to(opt.bounded(SPAN - 1.0, separation, SPAN + 1.0))
                cost += (separation - SPAN)**2
        opt.minimize(cost)
        opt.solver('ipopt', {'expand': True, 'print_time': False},
                   {'print_level': 0, 'sb': 'yes', 'max_iter': 60, 'tol': 1e-6})
        self.problem = opt
        self.guess = None

    def solve(self, initial, previous, refs, lead_plan=None):
        opt = self.problem
        opt.set_value(self.initial, initial)
        opt.set_value(self.previous, previous)
        opt.set_value(self.references, refs.T)
        if self.follower:
            opt.set_value(self.lead_plan, lead_plan.T)
        if self.guess is None:
            poses = np.vstack([np.column_stack([initial[:3], refs.T]),
                               np.full(STAGES + 1, initial[3])])
            actions = np.tile(previous[:, None], (1, STAGES))
        else:
            poses, actions = self.guess
        opt.set_initial(self.state, poses)
        opt.set_initial(self.action, actions)
        try:
            result = opt.solve()
        except RuntimeError:
            self.guess = None
            # Decelerate through the common actuator rate limit.
            return np.array([0.0, previous[1]]), np.tile(initial[:2], (STAGES, 1)), False
        poses, actions = result.value(self.state), result.value(self.action)
        self.guess = poses, actions
        return actions[:, 0], poses[:2, 1:].T, True


class DistributedMPC:
    def __init__(self, speed_limit):
        self.lead = MotionMPC(speed_limit)
        self.follow = MotionMPC(speed_limit, follower=True)
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
        refs_follow = follower_references(env.route, predicted)
        follow_cmd, _, follow_ok = self.follow.solve(initial[1], previous[1], refs_follow, predicted)
        self.failures += int(not lead_ok) + int(not follow_ok)

        actions = np.r_[lead_cmd, follow_cmd]
        actions[::2] = actions[::2] / self.lead.speed_limit - CONDITIONED_CRUISE
        actions[1::2] /= MAX_STEER
        return actions


def run_episode(scenario, seed, policy):
    env = CargoConvoy(scenario, delivery=True,
                      adaptive_speed=policy is not None and policy.observation_space.shape == (41,))
    observation, _ = env.reset(seed=1000 + seed)
    mpc = DistributedMPC(scenario.speed) if policy is None else None
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
    args = parser.parse_args()
    torch.set_num_threads(1)
    models = args.model or [ROOT / 'assets' / 'policy.zip']
    controllers = [('D-MPC', None), *((p.stem, PPO.load(p, device='cpu')) for p in models)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as file:
        writer = None
        for speed in args.speed or SPEEDS:
            for scenario in SPEED_SCENARIOS:
                for seed in (0, 1) if scenario.name == 'loose-crest' else (0,):
                    for label, policy in controllers:
                        row = dict(controller=label, **run_episode(replace(scenario, speed=speed), seed, policy))
                        if writer is None:
                            writer = csv.DictWriter(file, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        file.flush()
                        print(row, flush=True)


if __name__ == '__main__':
    main()
