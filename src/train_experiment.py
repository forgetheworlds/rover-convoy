"""Train predictive-slider PPO, SAC, or two learned rover actors."""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from environment import CargoConvoy
from learned_control import LeaderFollowerPolicy


class ForecastSliderConvoy(CargoConvoy):
    """Penalize outward slider motion before it becomes a large excursion."""

    def __init__(self):
        super().__init__(delivery=True, adaptive_speed=True, slider_weight=0.30)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        slide = info['slider']
        forecast = abs(slide + 0.45 * info['slide_speed'])
        added_excursion = max(0.0, forecast - max(0.08, abs(slide)))
        if np.isfinite(added_excursion):
            reward -= 0.30 * (added_excursion / 0.10)**2
        return obs, reward, terminated, truncated, info


def imitate_teacher(model, teacher, seed, sac):
    """Give the new architecture the same driving demonstrations."""
    env = ForecastSliderConvoy()
    states, commands = [], []
    rng = np.random.default_rng(seed)
    try:
        for _ in range(24):
            obs, _ = env.reset(seed=int(rng.integers(1_000_000, 2**31)))
            while True:
                action, _ = teacher.predict(obs, deterministic=True)
                states.append(obs)
                commands.append(action)
                obs, _, terminal, timeout, _ = env.step(action)
                if terminal or timeout:
                    break
    finally:
        env.close()
    inputs = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    targets = torch.as_tensor(np.asarray(commands), dtype=torch.float32)
    actor = model.policy.actor if sac else model.policy.mlp_extractor
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.002)
    for _ in range(20):
        for batch in torch.from_numpy(rng.permutation(len(states))).split(256):
            prediction = (actor(inputs[batch], deterministic=True) if sac
                          else actor.forward_actor(inputs[batch]))
            loss = torch.nn.functional.mse_loss(prediction, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f'Imitated the existing PPO on {len(states)} transitions.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('ppo', 'sac', 'leader-follower'))
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=196608)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    parser.add_argument('--seed', type=int, default=301)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    teacher = PPO.load(args.teacher, device='cpu')
    if teacher.observation_space.shape != (41,):
        parser.error('Teacher needs 41 inputs.')

    def factory(rank):
        return lambda: Monitor(ForecastSliderConvoy(), filename=str(args.out / f'env-{rank}'),
                               info_keywords=('success', 'stop_reason'))

    vectorize = SubprocVecEnv if args.workers > 2 else DummyVecEnv
    env = vectorize([factory(i) for i in range(args.workers)])
    try:
        if args.method == 'ppo':
            model = PPO.load(args.teacher, env=env, device='cpu', seed=args.seed,
                             custom_objects={'learning_rate': 5e-5})
        elif args.method == 'sac':
            model = SAC('MlpPolicy', env, seed=args.seed, device='cpu', learning_rate=1.5e-4,
                        batch_size=256, buffer_size=250_000, learning_starts=1024,
                        train_freq=(8, 'step'), gradient_steps=4, gamma=0.997, ent_coef=0.03,
                        policy_kwargs={'net_arch': {'pi': [64, 64], 'qf': [128, 128]},
                                       'activation_fn': torch.nn.Tanh}, verbose=1)
        else:
            model = PPO(LeaderFollowerPolicy, env, seed=args.seed, device='cpu',
                        n_steps=512, batch_size=256, n_epochs=6, gamma=0.997,
                        learning_rate=7e-5, policy_kwargs={'log_std_init': -2.4, 'ortho_init': False}, verbose=1)
        if args.method != 'ppo':
            imitate_teacher(model, teacher, args.seed, args.method == 'sac')
        model.learn(total_timesteps=args.steps, reset_num_timesteps=args.method != 'ppo',
                    log_interval=8, callback=CheckpointCallback(
                        save_freq=max(1, 65536 // args.workers), save_path=args.out, name_prefix='interim'))
        model.save(args.out / 'policy')
    finally:
        env.close()


if __name__ == '__main__':
    main()
