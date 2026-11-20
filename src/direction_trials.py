"""Run the field-envelope benchmark in the chosen travel direction."""

import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO
import torch

from benchmark import run_episode
from config import FIELD, SPEED
from field_pair import OBS_SIZE, FieldPairEnv
from vtr3_mpc import load_solvers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', type=Path)
    source.add_argument('--vtr3', type=Path)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--seed', type=int, action='append')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = PPO.load(args.model, device='cpu') if args.model else None
    solvers = load_solvers(args.vtr3) if args.vtr3 else None
    rows = []
    for scenario in FIELD:
        for seed in args.seed or (71, 91, 111):
            env = FieldPairEnv(scenario, reverse=args.reverse,
                               direction_input=policy is not None and policy.observation_space.shape[0] > OBS_SIZE)
            run = run_episode(scenario, seed, policy, solvers, mpc_cruise=SPEED, environment=env)
            row = dict(direction='reverse' if args.reverse else 'forward',
                       controller='learned' if policy else 'upstream-vtr3', **run)
            rows.append(row)
            print(row, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
