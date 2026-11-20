"""Choose a learned controller on development courses and evaluate a reserved suite."""

import argparse
from dataclasses import replace
from pathlib import Path
from statistics import mean
import csv

from stable_baselines3 import PPO, SAC
import torch

from benchmark import run_episode
from config import Scenario
from vtr3_mpc import REVISION, load_solvers

# Fixed before training finishes. Neither suite is sampled by a special-case
# rollout generator; training uses the environment's domain randomization.
DEVELOPMENT = (
    Scenario('dev-turns', amplitude=-1.37, frequency=0.295, distance=56, friction=0.62, pose_noise=.015),
    Scenario('dev-crest', amplitude=.62, frequency=.223, distance=64, slope_deg=6.8, crest_x=29),
    Scenario('dev-loose', amplitude=1.16, frequency=.257, distance=53, mass=552, friction=.45,
             slope_deg=8.1, crest_x=34, pose_noise=.021, state_delay=1),
    Scenario('dev-alternating', amplitude=-.95, frequency=.316, distance=60, mass=502,
             friction=.56, slope_deg=4.6, crest_x=42, pose_noise=.028, state_delay=2),
)

FIELD = (
    Scenario('field-envelope-crest', amplitude=-.61, frequency=.234, distance=60, mass=475,
             friction=.72, slope_deg=7.8, crest_x=34, speed=.75, pose_noise=.022, state_delay=1),
    Scenario('field-envelope-figure-eight', figure8=True, mass=475, friction=.74, slope_deg=7.8,
             crest_x=14, speed=.75, pose_noise=.016, state_delay=1),
)
STRESS = (
    Scenario('reserved-straight', distance=61, speed=3, mass=483, friction=.91),
    Scenario('reserved-bends', amplitude=1.32, frequency=.276, distance=58, speed=3, friction=.78),
    Scenario('reserved-crest', amplitude=-.72, frequency=.233, distance=64, speed=3, mass=475,
             friction=.69, slope_deg=7.8, crest_x=37),
    Scenario('reserved-loose', amplitude=.93, frequency=.274, distance=57, speed=3, mass=529,
             friction=.43, slope_deg=8.25, crest_x=32, pose_noise=.027, state_delay=1),
)


def evaluate(method, controller, scenarios, solvers=None):
    for case, seed, nominal in scenarios:
        metrics = run_episode(case, seed, controller, solvers, mpc_cruise=nominal)
        yield dict(controller=method, mpc_nominal_mps=nominal or case.speed,
                   vtr3_revision=REVISION, **metrics)


def score(rows):
    """Completion first, then peak slider error, time out of band and delivery time."""
    complete = [row for row in rows if row['success']]
    if not complete:
        return (0, float('inf'), float('inf'), float('inf'))
    return (-len(complete), max(row['slide_peak_cm'] for row in complete),
            mean(row['slider_over_10cm_pct'] for row in complete),
            mean(row['time_s'] for row in complete))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=('select', 'test'))
    parser.add_argument('--incumbent', required=True, type=Path)
    parser.add_argument('--ppo', type=Path)
    parser.add_argument('--sac', type=Path)
    parser.add_argument('--leader-follower', type=Path)
    parser.add_argument('--vtr3', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    incumbent = PPO.load(args.incumbent, device='cpu')
    if args.suite == 'select':
        if args.ppo is None or args.sac is None:
            parser.error('selection requires --ppo and --sac')
        controllers = {'incumbent': incumbent, 'slider-forecast-ppo': PPO.load(args.ppo, device='cpu'),
                       'sac': SAC.load(args.sac, device='cpu')}
        scenarios = [(replace(case, speed=speed), 27, None)
                     for case in DEVELOPMENT for speed in (1.5, 3.0)]
        scores = {}
        rows = []
        for name, controller in controllers.items():
            result = list(evaluate(name, controller, scenarios))
            scores[name] = score(result)
            rows.extend(result)
            print(name, scores[name], flush=True)
        winner = min(scores, key=scores.get)
        (args.out / 'selection.txt').write_text(winner + '\n')
        controllers[winner].save(args.out / 'selected')
    else:
        marker = args.out / 'selection.txt'
        if not marker.is_file() or args.leader_follower is None or args.vtr3 is None:
            parser.error('testing requires completed selection, --leader-follower and --vtr3')
        winner = marker.read_text().strip()
        algorithm = SAC if winner == 'sac' else PPO
        controllers = {'selected-' + winner: algorithm.load(args.out / 'selected.zip', device='cpu'),
                       'leader-follower': PPO.load(args.leader_follower, device='cpu'),
                       'upstream-vtr3': None}
        if winner != 'incumbent':
            controllers['incumbent'] = incumbent
        scenarios = [(case, 71, .50) for case in FIELD] + [(case, 73, None) for case in STRESS]
        solvers = load_solvers(args.vtr3)
        rows = []
        for name, controller in controllers.items():
            for r in evaluate(name, controller, scenarios, solvers):
                rows.append(r)
                print(r, flush=True)
    with (args.out / f'{args.suite}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
