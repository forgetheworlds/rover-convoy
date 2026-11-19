"""Evaluate learned and geometric controllers on identical scenario presets."""

import csv
from dataclasses import replace
import numpy as np

from config import DT, HELD_OUT, MAX_SPEED, SPEED, SPEED_SCENARIOS, SWEEP_SPEEDS
from control import geometric_action
from environment import CargoConvoy

def evaluate_episode(scenario, seed, policy, reference_speed, speed_sweep, *, delivery=False,
                     adaptive_speed=False):
    env = CargoConvoy(scenario, speed_conditioned=speed_sweep, delivery=delivery,
                      adaptive_speed=adaptive_speed)
    observation, _ = env.reset(seed=1000 + seed)
    observation_size = policy.observation_space.shape[0] if policy is not None else len(observation)
    start_m = env.metrics['progress'][1]
    speed_limit = env.speed_limit if speed_sweep else None
    errors, slide, tilt, pin = [], [], [], []
    cruise_speeds = []
    peak_pair_speed = 0.0
    try:
        while True:
            action = (policy.predict(observation[:observation_size], deterministic=True)[0] if policy else
                      geometric_action(observation, reference_speed, speed_limit=speed_limit))
            observation, _, terminated, truncated, info = env.step(action)
            errors.append(info['lateral'])
            slide.append(info['slider'])
            tilt.append(info['tilt'])
            pin.append(info['pin_error'])
            slower_speed = float(np.min(info['body_speed']))
            peak_pair_speed = max(peak_pair_speed, slower_speed)
            progress = info['progress'][1] - start_m
            if 20 <= progress <= 55:
                cruise_speeds.append(slower_speed)
            if terminated or truncated:
                break
    finally:
        env.close()
    errors = np.asarray(errors)
    row = dict(scenario=scenario.name, seed=seed, controller='ppo' if policy else 'geometric',
               success=int(info['success']), violation=int(info['constraint_violation']), stop_reason=info['stop_reason'],
               steps=len(errors), progress_m=round(info['progress'][1], 3),
               time_s=round(len(errors) * DT, 2),
               mean_progress_mps=round(progress / (len(errors) * DT), 4),
               rover_a_rmse_cm=round(100 * np.sqrt(np.mean(np.square(errors[:, 0]))), 3),
               rover_b_rmse_cm=round(100 * np.sqrt(np.mean(np.square(errors[:, 1]))), 3),
               slide_rmse_cm=round(100 * np.sqrt(np.mean(np.square(slide))), 3),
               slide_peak_cm=round(100 * np.max(np.abs(slide)), 3),
               slide_within_5cm_pct=round(100 * np.mean(np.abs(slide) <= 0.05), 2),
               tilt_peak_deg=round(np.rad2deg(np.max(tilt)), 3),
               pin_error_peak_cm=round(100 * np.max(pin), 3))
    if speed_sweep:
        cruise = np.asarray(cruise_speeds)
        row.update(target_speed_mps=speed_limit, peak_pair_speed_mps=round(peak_pair_speed, 3),
                   cruise_pair_speed_mps=round(float(cruise.mean()), 3) if cruise.size else '',
                   cruise_at_90pct_target_pct=round(100 * np.mean(cruise >= 0.9 * speed_limit), 2)
                   if cruise.size else '')
    return row


def evaluate(args):
    policy = None
    if args.model:
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        policy = PPO.load(args.model, device='cpu')
        expected = {(29,), (35,), (41,)} if args.speed_sweep else {(28,)}
        if policy.observation_space.shape not in expected:
            raise ValueError(f'Checkpoint observations {policy.observation_space.shape} are incompatible '
                             f'with this benchmark. Check --speed-sweep.')

    delivery = args.delivery_eval or (policy is not None and policy.observation_space.shape in ((35,), (41,)))
    scenarios = ([replace(scenario, speed=speed) for speed in SWEEP_SPEEDS for scenario in SPEED_SCENARIOS]
                 if args.speed_sweep else HELD_OUT)
    rows = []
    for scenario in scenarios:
        if args.short and scenario.figure8:
            continue
        reference_speed = scenario.speed if args.speed_sweep else args.reference_speed
        for seed in range(2 if args.speed_sweep else 3):
            for controller in (None, policy) if policy else (None,):
                row = evaluate_episode(scenario, seed, controller, reference_speed, args.speed_sweep,
                                       delivery=delivery,
                                        adaptive_speed=bool(policy is not None and policy.observation_space.shape == (41,)))
                print(row, flush=True)
                rows.append(row)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with args.out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
