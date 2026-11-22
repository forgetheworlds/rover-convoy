"""Command-line entry point for training and evaluating rover policies.

    python src/convoy.py train --delivery --adaptive-speed \\
        --initialize-from assets/policy.zip --workers 4 --steps 524288
"""

import argparse
from math import isfinite
from pathlib import Path

from config import SPEED, MAX_SPEED, ROOT
from environment import DELIVERY_SLIDER_WEIGHT
from evaluation import evaluate
from training import train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train')
    run.add_argument('--steps', type=int, default=131_072)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--out', type=Path, default=ROOT / 'runs' / 'models')
    initialization = run.add_mutually_exclusive_group()
    initialization.add_argument('--from-scratch', action='store_true')
    initialization.add_argument('--resume', type=Path, help='continue PPO from a matching checkpoint')
    initialization.add_argument('--initialize-from', type=Path, help='begin with actor weights from a PPO checkpoint')
    run.add_argument('--slider-weight', type=float, default=DELIVERY_SLIDER_WEIGHT,
                     help='delivery penalty per 5 cm of absolute slider displacement')
    run.add_argument('--slider-tail-weight', type=float, default=0.0,
                     help='quadratic penalty coefficient above 10 cm of slider displacement')
    run.add_argument('--speed-conditioned', action='store_true',
                     help='train one policy over several speed targets; reserve 3 m/s for evaluation')
    run.add_argument('--delivery', action='store_true', help='train for whole-route delivery at up to 3 m/s')
    run.add_argument('--adaptive-speed', action='store_true',
                     help='train 41-input policy with future curvature preview; no external pacing')
    run.add_argument('--workers', type=int, choices=range(1, 9), default=2)
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
    if args.command == 'train':
        if not isfinite(args.slider_weight) or args.slider_weight < 0:
            parser.error('--slider-weight must be finite and nonnegative')
        if not isfinite(args.slider_tail_weight) or args.slider_tail_weight < 0:
            parser.error('--slider-tail-weight must be finite and nonnegative')
        if (args.slider_weight != DELIVERY_SLIDER_WEIGHT or args.slider_tail_weight) and not args.delivery:
            parser.error('slider weight overrides require --delivery')
    if args.command == 'evaluate' and not 0 < args.reference_speed <= MAX_SPEED:
        parser.error(f'--reference-speed must be between 0 and {MAX_SPEED} m/s')
    if args.command == 'evaluate' and args.delivery_eval and not args.speed_sweep:
        parser.error('--delivery-eval is part of --speed-sweep')
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
