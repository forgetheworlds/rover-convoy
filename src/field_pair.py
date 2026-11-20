"""Train both learned rover roles in the paper's nominal field-speed regime."""

import argparse
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from config import CONDITIONED_CRUISE, MAX_SPEED, MAX_STEER, SLIDER_LIMIT, SPAN, SPEED, WHEELBASE, Scenario
from control import geometric_action
from environment import CargoConvoy
from learned_control import ConvoyActors, LeaderFollowerPolicy
from leader_preview import PREVIEW_FEATURES, rollout_features
from routes import route_poses

OBS_SIZE = 41 + PREVIEW_FEATURES


class FieldPairPolicy(LeaderFollowerPolicy):
    def _build_mlp_extractor(self):
        if self.features_dim not in (OBS_SIZE, OBS_SIZE + 1):
            raise ValueError(f'Field role policy requires {OBS_SIZE} or {OBS_SIZE + 1} observations.')
        self.mlp_extractor = ConvoyActors(self.features_dim - 41,
                                         direction_input=self.features_dim > OBS_SIZE).to(self.device)


class FieldPairEnv(gym.Wrapper):
    """Publish the last learned leader command's forecast at each control tick."""

    def __init__(self, scenario=None, *, slider_weight=0.32, speed_weight=1.5,
                 reverse=False, mixed_direction=False, direction_input=False, model=None):
        super().__init__(CargoConvoy(scenario, model=model, delivery=True, adaptive_speed=True,
                                    slider_weight=slider_weight, reverse=reverse))
        self.fixed = scenario
        self.mixed_direction = mixed_direction
        self.direction_input = direction_input
        self.rng = np.random.default_rng()
        self.speed_weight = speed_weight
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE + int(direction_input),), np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.mixed_direction:
            self.env.reverse = bool(self.rng.integers(2))
        if self.fixed is None:
            rng = self.rng
            self.env.fixed_scenario = Scenario(
                name='field-training', speed=MAX_SPEED, amplitude=rng.uniform(-1.7, 1.7),
                frequency=rng.uniform(.19, .34), distance=float(rng.uniform(21, 45)),
                friction=rng.uniform(.46, .98), mass=rng.uniform(415, 525),
                slope_deg=float(rng.choice([0., rng.uniform(3, 9)])),
                crest_x=rng.uniform(12, 27), pose_noise=rng.uniform(0, .028),
                state_delay=int(rng.integers(0, 3)),
            )
            if self.direction_input and self.mixed_direction and self.env.reverse and rng.random() < .20:
                self.env.fixed_scenario = replace(self.env.fixed_scenario, figure8=True,
                                                  loop_width=rng.uniform(20., 26.),
                                                  loop_height=rng.uniform(10., 13.))
        observation, info = self.env.reset(seed=seed, options=options)
        self.leader_index = int(np.searchsorted(self.env.route.s, SPAN))
        return self._observe(observation, [SPEED / MAX_SPEED - CONDITIONED_CRUISE, 0.]), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        leader_speed_error = (info['body_speed'][0] - SPEED) / .25
        reward -= self.speed_weight * leader_speed_error ** 2
        return self._observe(observation, action[:2]), reward, terminated, truncated, info

    def _observe(self, observation, leader_action):
        preview, self.leader_index = rollout_features(self.env, leader_action, self.leader_index)
        value = np.r_[observation, preview]
        if self.direction_input:
            value = np.r_[value, -1. if self.env.reverse else 1.]
        return value.astype(np.float32)


def teacher_action(observation, env=None):
    """Demonstration controller used for initialization only."""
    action = geometric_action(observation, SPEED, speed_limit=MAX_SPEED)
    action[0] = SPEED / MAX_SPEED - CONDITIONED_CRUISE
    slider = observation[18] * SLIDER_LIMIT
    slider_velocity = observation[19] * .5
    lead_speed = observation[2] * MAX_SPEED
    follow_speed = .75 * lead_speed + .25 * SPEED - 1.55 * slider - .45 * slider_velocity
    action[2] = follow_speed / MAX_SPEED - CONDITIONED_CRUISE
    if env is not None and env.env.reverse and env.direction_input:
        # Track using the effective fixed-wheel axle, ahead of the body frame
        # when backing, so the teacher compensates for the steering geometry.
        route = env.env.route
        for i, pose in enumerate(env.env.estimated_states):
            cos_yaw, sin_yaw = np.cos(pose[2]), np.sin(pose[2])
            axle = pose[:2] + .61 * np.array([cos_yaw, sin_yaw])
            _, station, _ = route.track(axle, int(env.env.observed_indices[i]))
            target = route_poses(route, np.array([station + 1.7]))[0, :2]
            delta = target - axle
            lateral = -sin_yaw * delta[0] + cos_yaw * delta[1]
            action[2 * i + 1] = np.arctan2(2 * WHEELBASE * lateral, delta @ delta) / MAX_STEER
    return np.clip(action, -1., 1.).astype(np.float32)


def copy_role_weights(model, source):
    current = model.policy.mlp_extractor
    previous = source.policy.mlp_extractor
    with torch.no_grad():
        for actor, prior in ((current.leader, previous.leader), (current.follower, previous.follower)):
            width = prior[0].in_features
            actor[0].weight.zero_()
            actor[0].weight[:, :width].copy_(prior[0].weight)
            actor[0].bias.copy_(prior[0].bias)
            actor[2:].load_state_dict(prior[2:].state_dict())


def imitate(model, seed, *, mixed_direction=False, direction_input=False):
    """Fit both actors on expert trajectories and on states the actors visit."""
    env = FieldPairEnv(mixed_direction=mixed_direction, direction_input=direction_input)
    rng = np.random.default_rng(seed)
    actor = model.policy.mlp_extractor
    optimizer = torch.optim.Adam([*actor.leader.parameters(), *actor.follower.parameters()], lr=0.001)
    states, labels = [], []
    try:
        for rounds, learned_chance in ((28, 0.), (12, .6), (12, .85)):
            completed = 0
            for _ in range(rounds):
                observation, _ = env.reset(seed=int(rng.integers(1_000_000, 2**31)))
                while True:
                    teacher = teacher_action(observation, env)
                    states.append(observation)
                    labels.append(teacher)
                    action = (model.predict(observation, deterministic=True)[0]
                              if rng.random() < learned_chance else teacher)
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        completed += int(info['success'])
                        break
            inputs = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            commands = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
            for _ in range(18):
                for batch in torch.from_numpy(rng.permutation(len(states))).split(256):
                    output = actor.forward_actor(inputs[batch])
                    loss = torch.nn.functional.mse_loss(output, commands[batch])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print(f'Demonstration round {learned_chance:g}: {completed}/{rounds} completed; '
                  f'{len(states)} labeled states.', flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--warm-from', type=Path, help='original 41-input learned pair')
    source.add_argument('--adapt-from', type=Path, help='fit the existing field pair in both directions')
    source.add_argument('--resume', type=Path, help='field pair to fine-tune')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=65536)
    parser.add_argument('--seed', type=int, default=971)
    parser.add_argument('--slider-weight', type=float, default=.32)
    parser.add_argument('--mixed-direction', action='store_true', help='train in the direction-of-travel frame in both directions')
    parser.add_argument('--direction-input', action='store_true', help='let both actors distinguish forward and reverse dynamics')
    parser.add_argument('--speed-weight', type=float, default=1.5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.warm_from or args.adapt_from:
        env = FieldPairEnv(mixed_direction=args.mixed_direction, direction_input=args.direction_input)
        try:
            if args.adapt_from and not args.direction_input:
                model = PPO.load(args.adapt_from, env=env, device='cpu', seed=args.seed)
            else:
                model = PPO(FieldPairPolicy, env, device='cpu', seed=args.seed, n_steps=256,
                        batch_size=256, n_epochs=4, gamma=.997, learning_rate=1.5e-5,
                        target_kl=.009, policy_kwargs={'ortho_init': False, 'log_std_init': -3.5})
                copy_role_weights(model, PPO.load(args.warm_from or args.adapt_from, device='cpu'))
            imitate(model, args.seed, mixed_direction=args.mixed_direction, direction_input=args.direction_input)
            model.save(args.out / 'policy')
        finally:
            env.close()
        return

    env = SubprocVecEnv([lambda: Monitor(FieldPairEnv(slider_weight=args.slider_weight,
                                                    speed_weight=args.speed_weight,
                                                    mixed_direction=args.mixed_direction,
                                                    direction_input=args.direction_input),
                                        info_keywords=('success', 'stop_reason')) for _ in range(4)])
    try:
        model = PPO.load(args.resume, env=env, seed=args.seed, device='cpu', verbose=1)
        model.learn(args.steps, log_interval=8,
                    callback=CheckpointCallback(save_freq=4096, save_path=args.out))
        model.save(args.out / 'policy')
    finally:
        env.close()


if __name__ == '__main__':
    main()
