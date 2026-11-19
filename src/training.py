"""PPO warm starts, actor transfer, and multi-stage training."""

from pathlib import Path
import numpy as np

from control import geometric_action
from environment import CargoConvoy

def warm_start_policy(model, seed, speed_conditioned, delivery=False, adaptive_speed=False):
    """Supervise the actor on geometric rollouts before online PPO updates."""
    import torch

    env = CargoConvoy(speed_conditioned=speed_conditioned, delivery=delivery,
                      adaptive_speed=adaptive_speed)
    rng = np.random.default_rng(seed)
    states, commands = [], []
    try:
        for _ in range(24):
            obs, _ = env.reset(seed=int(rng.integers(1_000_000, 2**31)))
            cruise = env.speed_limit * rng.uniform(0.86, 0.98) if speed_conditioned else rng.uniform(0.58, 0.70)
            while True:
                speed_limit = env.speed_limit if speed_conditioned else None
                action = np.clip(geometric_action(obs, cruise, speed_limit=speed_limit), -1, 1)
                states.append(obs)
                commands.append(action)
                obs, _, terminal, timeout, _ = env.step(action)
                if terminal or timeout:
                    break
    finally:
        env.close()

    inputs = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    targets = torch.as_tensor(np.asarray(commands), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.002)
    for _ in range(15):
        for indices in torch.from_numpy(rng.permutation(len(states))).split(256):
            predicted = model.policy.get_distribution(inputs[indices]).distribution.mean
            loss = torch.nn.functional.mse_loss(predicted, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f'Warm-started on {len(states)} simulated geometric transitions.', flush=True)


def initialize_actor(model, checkpoint):
    """Transfer only the actor and initialize extra input weights to zero."""
    import torch
    from stable_baselines3 import PPO

    prior = PPO.load(checkpoint, device='cpu').policy.state_dict()
    actor_params = ('mlp_extractor.policy_net.', 'action_net.')
    with torch.no_grad():
        for name, target in model.policy.state_dict().items():
            if name != 'log_std' and not name.startswith(actor_params):
                continue
            weight = prior[name]
            if target.shape == weight.shape:
                target.copy_(weight)
            elif (name == 'mlp_extractor.policy_net.0.weight'
                  and weight.shape[0] == target.shape[0] and weight.shape[1] < target.shape[1]):
                target.zero_()
                target[:, :weight.shape[1]].copy_(weight)
            else:
                raise ValueError(f'Cannot transfer actor parameter {name}: {weight.shape} → {target.shape}')
    print(f'Initialized actor from {checkpoint}; training a fresh critic.', flush=True)


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.out.mkdir(exist_ok=True, parents=True)
    speed_conditioned = args.speed_conditioned or args.delivery

    def environment(rank):
        return lambda: Monitor(CargoConvoy(speed_conditioned=speed_conditioned,
                                             delivery=args.delivery, adaptive_speed=args.adaptive_speed),
                               filename=str(args.out / f'worker-{rank}') if args.delivery else None,
                               info_keywords=('success', 'stop_reason') if args.delivery else ())

    vectorize = SubprocVecEnv if args.workers > 2 else DummyVecEnv
    env = vectorize([environment(rank) for rank in range(args.workers)])
    ppo = PPO('MlpPolicy', env, device='cpu', seed=args.seed, n_steps=512,
              batch_size=256 if args.delivery else 128, n_epochs=6 if args.delivery else 8,
              gamma=0.997 if args.delivery else 0.995,
              learning_rate=(lambda remaining: 3e-5 + 1.2e-4 * remaining) if args.delivery else 1.5e-4,
              policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}, 'log_std_init': -2.4},
              verbose=1)
    try:
        if args.initialize_from:
            initialize_actor(ppo, args.initialize_from)
        elif not args.from_scratch:
            warm_start_policy(ppo, args.seed, speed_conditioned, args.delivery, args.adaptive_speed)
            ppo.save(args.out / 'warm-start')
        ppo.learn(total_timesteps=args.steps,
                  log_interval=4 if args.delivery else 1,
                  callback=CheckpointCallback(save_freq=(65536 if args.delivery else 32768) // args.workers,
                                              save_path=args.out, name_prefix='interim'))
        ppo.save(args.out / 'ppo')
    finally:
        env.close()
