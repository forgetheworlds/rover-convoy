"""Compare the learned joint controller with the upstream VT&R3 D-MPC solvers."""

from time import perf_counter_ns

import numpy as np

from config import (
    CONDITIONED_CRUISE, DT, MAX_STEER, SPAN, SLEW, WHEELBASE,
)
from environment import CargoConvoy, SLIDER_TARGET
from routes import angle_wrap, follower_references, projected_distance, route_poses

from vtr3_mpc import MotionMPC, STAGES, STEP

def reference_progress(initial_speed, cruise):
    """Reference distance at each stage with a slew-feasible cruise ramp."""
    times = STEP * np.arange(1, STAGES + 1)
    acceleration = SLEW[0] / DT
    ramp = np.minimum(times, abs(cruise - initial_speed) / acceleration)
    distance = initial_speed * ramp + np.sign(cruise - initial_speed) * acceleration * ramp**2 / 2
    return distance + cruise * (times - ramp)


class DistributedMPC:
    def __init__(self, speed_limit, solvers, *, reverse=False, control_dt=DT, span=SPAN):
        self.speed_limit = speed_limit
        self.span = float(span)
        self.lead = MotionMPC(solvers[0], speed_limit, reverse=reverse, control_dt=control_dt, span=span)
        self.follow = MotionMPC(solvers[1], speed_limit, follower=True, reverse=reverse, control_dt=control_dt, span=span)
        self.failures = 0
        self.leader_index = None

    def __call__(self, env, cruise):
        estimated = env.estimated_states.copy()
        yaw, speed, steer = estimated[:, 2], estimated[:, 3], estimated[:, 4]
        change = env.estimate_age * speed * np.tan(steer) / WHEELBASE
        estimated[:, 0] += env.estimate_age * speed * np.cos(yaw + change / 2)
        estimated[:, 1] += env.estimate_age * speed * np.sin(yaw + change / 2)
        estimated[:, 2] = angle_wrap(yaw + change)
        initial = estimated[:, [0, 1, 2, 4]]
        previous = env.commands.reshape(2, 2)
        if env.route.figure8:
            if self.leader_index is None:
                self.leader_index = int(np.searchsorted(env.route.s, self.span))
            self.leader_index, start, _ = env.route.track(initial[0, :2], self.leader_index)
        else:
            start = projected_distance(env.route, initial[0:1, :2])[0]
        distances = start + reference_progress(previous[0, 0], cruise)
        refs = route_poses(env.route, distances)

        lead_cmd, predicted, lead_ok = self.lead.solve(initial[0], previous[0], refs)
        lead_hint = None
        if env.route.figure8:
            lead_xy = np.vstack((initial[0, :2], predicted[:, :2]))
            lead_hint = start + np.cumsum(np.linalg.norm(np.diff(lead_xy, axis=0), axis=1))
        refs_follow = follower_references(env.route, predicted[:, :2], lead_distance_hint=lead_hint)
        follow_cmd, _, follow_ok = self.follow.solve(initial[1], previous[1], refs_follow, predicted)
        self.failures += int(not lead_ok) + int(not follow_ok)

        actions = np.r_[lead_cmd, follow_cmd]
        actions[::2] = actions[::2] / self.speed_limit - CONDITIONED_CRUISE
        actions[1::2] /= MAX_STEER
        return actions


def run_episode(scenario, seed, policy, solvers, *, mpc_cruise=None, environment=None):
    cruise = scenario.speed if mpc_cruise is None else mpc_cruise
    if not 0 < cruise <= scenario.speed:
        raise ValueError("MPC cruise must be positive and below or equal to the speed ceiling.")
    env = environment if environment is not None else CargoConvoy(
        scenario, delivery=True, adaptive_speed=policy is not None and policy.observation_space.shape == (41,))
    observation, _ = env.reset(seed=1000 + seed)
    control_dt = env.unwrapped.control_dt
    mpc = (DistributedMPC(scenario.speed, solvers, reverse=env.unwrapped.reverse,
                          control_dt=control_dt, span=env.unwrapped.span)
           if policy is None else None)
    slide, cross, elapsed = [], [], []
    try:
        while True:
            start = perf_counter_ns()
            action = (mpc(env.unwrapped, cruise) if mpc else policy.predict(observation, deterministic=True)[0])
            elapsed.append((perf_counter_ns() - start) * 1e-6 + getattr(env, 'controller_preparation_ms', 0.0))
            observation, _, ended, timed_out, info = env.step(action)
            slide.append(info['slider'])
            cross.append(info['lateral'])
            if ended or timed_out:
                break
    finally:
        env.close()
    lateral_rms = np.sqrt(np.mean(np.square(cross), axis=0))
    above_target = int(np.count_nonzero(np.abs(slide) > SLIDER_TARGET))
    return dict(
        course=scenario.name, speed=scenario.speed, seed=seed,
        success=int(info['success']), stop=info['stop_reason'],
        progress_m=round(float(info['progress'][1]), 3), time_s=round(len(slide) * control_dt, 2),
        slide_rmse_cm=round(100 * np.sqrt(np.mean(np.square(slide))), 3),
        slide_peak_cm=round(100 * np.max(np.abs(slide)), 3),
        slider_over_10cm_s=round(above_target * control_dt, 2),
        slider_over_10cm_pct=round(100 * above_target / len(slide), 2),
        rover_a_rmse_cm=round(100 * lateral_rms[0], 3), rover_b_rmse_cm=round(100 * lateral_rms[1], 3),
        path_peak_cm=round(100 * np.max(np.abs(cross)), 3),
        controller_p95_ms=round(float(np.quantile(elapsed, 0.95)), 4),
        controller_deadline_misses=int(np.count_nonzero(np.asarray(elapsed) > control_dt * 1000)),
        control_steps=len(elapsed),
        solver_failures=mpc.failures if mpc else 0,
    )
