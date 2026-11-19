"""Command-line entry point for training and evaluating rover policies.

    python src/convoy.py train --delivery --adaptive-speed \\
        --initialize-from assets/policy.zip --workers 4 --steps 524288
"""

import argparse
from pathlib import Path

from config import SPEED, ROOT
from evaluation import evaluate
from training import train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train')
    run.add_argument('--steps', type=int, default=131_072)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--out', type=Path, default=ROOT / 'runs' / 'models')
    run.add_argument('--from-scratch', action='store_true')
    run.add_argument('--speed-conditioned', action='store_true',
                     help='train one policy over several speed targets; reserve 3 m/s for evaluation')
    run.add_argument('--delivery', action='store_true', help='train for whole-route delivery at up to 3 m/s')
    run.add_argument('--adaptive-speed', action='store_true',
                     help='train 41-input policy with future curvature preview; no external pacing')
    run.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    run.add_argument('--initialize-from', type=Path, help='begin with actor weights from a PPO checkpoint')
    run = sub.add_parser('evaluate')
    run.add_argument('--model', type=Path)
    run.add_argument('--short', action='store_true')
    run.add_argument('--speed-sweep', action='store_true', help='evaluate 60 m routes at 0.5–3 m/s')
    run.add_argument('--delivery-eval', action='store_true',
                     help='give the old policy and geometric controller the same pose estimates as the delivery policy')
    run.add_argument('--reference-speed', type=float, default=SPEED,
                     help='geometric controller cruise target in m/s')
    run.add_argument('--out', type=Path, default=ROOT / 'runs' / 'policy-evaluation.csv')
    args = parser.parse_args()
    if args.command == 'train' and args.adaptive_speed and not args.delivery:
        parser.error('--adaptive-speed requires --delivery')
    if args.command == 'train' and args.from_scratch and args.initialize_from:
        parser.error('--from-scratch and --initialize-from are mutually exclusive')
    if args.command == 'evaluate' and not 0 < args.reference_speed <= MAX_SPEED:
        parser.error(f'--reference-speed must be between 0 and {MAX_SPEED} m/s')
    if args.command == 'evaluate' and args.delivery_eval and not args.speed_sweep:
        parser.error('--delivery-eval is part of --speed-sweep')
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
