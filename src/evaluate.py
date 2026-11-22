"""Head-to-head evaluation of the pose-only learned policy vs. upstream VT&R3 D-MPC.

Suites:
  field    Forward & reverse crest and two-lap figure-eight (config.FIELD, 3 seeds)
  docking  Forward -> stop -> role swap -> reverse docking (2 scenarios)
  heldout  20 unseen in-distribution crest courses at 10 Hz
  shifted  20 out-of-distribution courses (heavier load, lower friction, noisier pose)
  speeds   0.75, 1.5, and 3.0 m/s speed caps at 50 Hz (with real-time deadline counts)
  spans    3.5, 4.15, and 5.5 m cargo spans on the field crest (3 seeds each)
  all      Run every suite above and write CSVs to --out-dir

Example:
  PYTHONPATH=src python src/evaluate.py --suite all --out-dir results
"""

import argparse
import csv
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from config import FIELD, MAX_SPEED, SPEED
from data import Scenario

_CACHE = {}


def make_courses(kind, count, seed=20261):
    """Deterministic crest courses; `shifted` leaves the training ranges on purpose."""
    rng = np.random.default_rng(seed + (0 if kind == 'heldout' else 1))
    courses = []
    for i in range(count):
        shifted = kind == 'shifted'
        courses.append(Scenario(
            name=f'{kind}-{i:02d}', speed=MAX_SPEED,
            amplitude=rng.choice([-1, 1]) * rng.uniform(1.75, 2.3) if shifted else rng.uniform(-1.7, 1.7),
            frequency=rng.uniform(.19, .34), distance=float(rng.uniform(21, 45)),
            friction=rng.uniform(.35, .44) if shifted else rng.uniform(.46, .98),
            mass=rng.uniform(535, 600) if shifted else rng.uniform(415, 525),
            slope_deg=float(rng.choice([0., rng.uniform(3, 9)])), crest_x=rng.uniform(12, 27),
            pose_noise=rng.uniform(.03, .045) if shifted else rng.uniform(0, .028),
            state_delay=int(rng.integers(0, 3))))
    return courses


def bootstrap_ci(values, rng, draws=4000):
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, (draws, len(values))).mean(axis=1)
    return np.quantile(means, [.025, .975])


def _run_job(job):
    import torch
    from stable_baselines3 import PPO
    from benchmark import run_episode
    from field_pair import OBS_SIZE, FieldPairEnv
    from vtr3_mpc import load_solvers
    torch.set_num_threads(1)

    controller, model_path, vtr3_path, scenario, seed, hz, reverse, cruise = job
    policy = _CACHE.setdefault(model_path, PPO.load(model_path, device='cpu')) if controller == 'learned' else None
    solvers = _CACHE.setdefault('solvers', load_solvers(Path(vtr3_path))) if controller == 'mpc' else None
    env = FieldPairEnv(
        scenario, reverse=reverse, control_hz=hz, profile=policy is not None,
        observation='pose-only' if policy is not None else 'full',
        direction_input=policy is not None and policy.observation_space.shape[0] > OBS_SIZE,
    )
    row = run_episode(scenario, seed, policy, solvers, mpc_cruise=cruise, environment=env)
    return dict(
        controller='learned-pose-only' if policy else 'vtr3-mpc',
        direction='reverse' if reverse else 'forward',
        span_m=scenario.span, control_hz=hz, cruise_mps=round(cruise, 3),
        mean_progress_mps=round(row['progress_m'] / row['time_s'], 4), **row,
    )


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _summarize(title, rows, group_keys):
    rng = np.random.default_rng(0)
    print(f'\n=== {title} ===')
    for key in dict.fromkeys(tuple(r[k] for k in group_keys) for r in rows):
        g = [r for r in rows if tuple(r[k] for k in group_keys) == key]
        tag = ' / '.join(str(v) for v in key)
        ok = f"{sum(int(r['success']) for r in g)}/{len(g)}"
        if 'slide_rmse_cm' in g[0]:
            rmse = [r['slide_rmse_cm'] for r in g]
            lo, hi = bootstrap_ci(rmse, rng)
            peak = np.mean([r['slide_peak_cm'] for r in g])
            path = np.mean([0.5 * (r['rover_a_rmse_cm'] + r['rover_b_rmse_cm']) for r in g])
            p95 = np.mean([r['controller_p95_ms'] for r in g])
            miss = sum(r['controller_deadline_misses'] for r in g)
            print(f'  {tag:<44} {ok:>5} | slide {np.mean(rmse):5.2f} [{lo:5.2f}, {hi:5.2f}] | peak {peak:4.1f} | path {path:5.2f} | p95 {p95:5.2f} ms | misses {miss}')
        else:
            peak = np.mean([r['slide_peak_cm'] for r in g])
            print(f'  {tag:<44} {ok:>5} | peak {peak:5.2f} cm')


def run_suite(name, model_path, vtr3_path, out_dir, workers):
    if name == 'docking':
        import torch
        from stable_baselines3 import PPO
        from maneuver import run_maneuver
        from vtr3_mpc import load_solvers
        torch.set_num_threads(1)
        policy = PPO.load(model_path, device='cpu')
        solvers = load_solvers(Path(vtr3_path))
        cases = (
            Scenario('curved-approach', amplitude=.54, frequency=.21, distance=24, speed=.75),
            Scenario('graded-approach', amplitude=-.63, frequency=.23, distance=26, speed=.75,
                     slope_deg=4.2, crest_x=17., friction=.67, pose_noise=.015, state_delay=1),
        )
        rows = []
        for sc in cases:
            for ctrl, pol, slv in (('learned-pose-only', policy, None), ('vtr3-mpc', None, solvers)):
                rows.append(dict(controller=ctrl, scenario=sc.name,
                                 **run_maneuver(sc, 1221, pol, slv, observation='pose-only')))
        _write_csv(out_dir / 'docking.csv', rows)
        _summarize('docking', rows, ('controller',))
        return

    jobs = []
    if name == 'field':
        for ctrl in ('learned', 'mpc'):
            for rev in (False, True):
                for sc in FIELD:
                    for seed in (71, 91, 111):
                        jobs.append((ctrl, str(model_path), str(vtr3_path), sc, seed, 10, rev, SPEED))
        group_keys = ('direction', 'course', 'controller')
    elif name in ('heldout', 'shifted'):
        courses = make_courses(name, 20)
        for ctrl in ('learned', 'mpc'):
            for i, sc in enumerate(courses):
                jobs.append((ctrl, str(model_path), str(vtr3_path), sc, 71 + i, 10, False, SPEED))
        group_keys = ('controller',)
    elif name == 'speeds':
        for ctrl in ('learned', 'mpc'):
            for cap in (0.75, 1.5, 3.0):
                sc = replace(FIELD[0], name=f'short-crest-{cap:g}', distance=39., crest_x=22., speed=cap)
                jobs.append((ctrl, str(model_path), str(vtr3_path), sc, 71, 50, False, cap * 2 / 3))
        group_keys = ('speed', 'controller')
    elif name == 'spans':
        for ctrl in ('learned', 'mpc'):
            for span in (3.5, 4.15, 5.5):
                sc = replace(FIELD[0], name=f'crest-span-{span:g}m', span=span)
                for seed in (71, 91, 111):
                    jobs.append((ctrl, str(model_path), str(vtr3_path), sc, seed, 10, False, SPEED))
        group_keys = ('span_m', 'controller')
    else:
        raise ValueError(f'Unknown suite: {name}')

    with Pool(workers) as pool:
        rows = list(pool.imap(_run_job, jobs))
    _write_csv(out_dir / f'{name}.csv', rows)
    _summarize(name, rows, group_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', choices=('field', 'docking', 'heldout', 'shifted', 'speeds', 'spans', 'all'), default='all')
    parser.add_argument('--model', type=Path, default=Path('assets/pose-only-policy.zip'))
    parser.add_argument('--vtr3', type=Path, default=Path('runs/vtr3'))
    parser.add_argument('--out-dir', type=Path, default=Path('results'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    suites = ('field', 'docking', 'heldout', 'shifted', 'speeds', 'spans') if args.suite == 'all' else (args.suite,)
    for suite in suites:
        run_suite(suite, args.model, args.vtr3, args.out_dir, args.workers)


if __name__ == '__main__':
    main()
