"""Train both learned rover roles in the paper's nominal field-speed regime."""

import argparse
from time import perf_counter_ns
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
from routes import angle_wrap, route_poses

OBS_SIZE = 41 + PREVIEW_FEATURES

# Positional inputs hidden from the learned actors so they see no more than the MPC does.
# Indices follow data.observation_names('adaptive_speed').
#   no-slider: zeroes the physical cargo slider position and speed.
#   pose-only: replaces slider position/speed with 2D Euclidean spacing & rate derived
#              strictly from the delayed noisy rover poses (identical to VT&R3 D-MPC),
#              replaces relative heading sin with noisy pose headings, and zeroes all
#              non-MPC channels (lateral velocity, yaw rate, 3D grade, roll, pitch).
HIDDEN_INPUTS = {
    'full': (),
    'no-slider': (18, 19),
    'pose-only': (3, 4, 12, 13, 20, 21, 31, 34),
}


def pose_derived_spacing(estimated_states, estimate_age, span=SPAN):
    """Estimate cargo spacing error and closing rate strictly from noisy 2D rover poses."""
    yaw, speed, steer = estimated_states[:, 2], estimated_states[:, 3], estimated_states[:, 4]
    dtheta = estimate_age * speed * np.tan(steer) / WHEELBASE
    x = estimated_states[:, 0] + estimate_age * speed * np.cos(yaw + dtheta / 2)
    y = estimated_states[:, 1] + estimate_age * speed * np.sin(yaw + dtheta / 2)
    heading = angle_wrap(yaw + dtheta)
    dx, dy = x[0] - x[1], y[0] - y[1]
    dist = max(float(np.hypot(dx, dy)), 1e-6)
    vx, vy = speed * np.cos(heading), speed * np.sin(heading)
    d_dot = (dx * (vx[0] - vx[1]) + dy * (vy[0] - vy[1])) / dist
    return -(dist - span), -d_dot, float(np.sin(heading[0] - heading[1]))


class FieldPairPolicy(LeaderFollowerPolicy):
    def _build_mlp_extractor(self):
        if self.features_dim not in (OBS_SIZE, OBS_SIZE + 1):
            raise ValueError(f'Field role policy requires {OBS_SIZE} or {OBS_SIZE + 1} observations.')
        self.mlp_extractor = ConvoyActors(self.features_dim - 41,
                                         direction_input=self.features_dim > OBS_SIZE).to(self.device)


class FieldPairEnv(gym.Wrapper):
    """Publish the last learned leader command's forecast at each control tick."""

    def __init__(self, scenario=None, *, slider_weight=0.32, speed_weight=1.5,
                 reverse=False, mixed_direction=False, direction_input=False, model=None, control_hz=10, profile=False,
                 observation='full', multi_speed=False):
        super().__init__(CargoConvoy(scenario, model=model, delivery=True, adaptive_speed=True,
                                    slider_weight=slider_weight, reverse=reverse, control_hz=control_hz))
        if observation not in HIDDEN_INPUTS:
            raise ValueError(f'Unknown observation set {observation!r}; choose from {sorted(HIDDEN_INPUTS)}.')
        self.observation_set = observation
        self.hidden = np.asarray(HIDDEN_INPUTS[observation], dtype=int)
        self.full_observation = None
        self.multi_speed = multi_speed
        self.profile = profile
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
            speed_cap = float(rng.choice((0.75, 0.75, 0.75, 1.5, 2.5, 3.0))) if self.multi_speed else MAX_SPEED
            self.env.fixed_scenario = Scenario(
                name='field-training', speed=speed_cap,
                amplitude=rng.uniform(-1.9, 1.9) if self.multi_speed else rng.uniform(-1.7, 1.7),
                frequency=rng.uniform(.19, .34), distance=float(rng.uniform(21, 45)),
                friction=rng.uniform(.40, 1.00) if self.multi_speed else rng.uniform(.46, .98),
                mass=rng.uniform(380, 580) if self.multi_speed else rng.uniform(415, 525),
                slope_deg=float(rng.choice([0., rng.uniform(3, 9)])),
                crest_x=rng.uniform(12, 27),
                pose_noise=rng.uniform(0, .038) if self.multi_speed else rng.uniform(0, .028),
                state_delay=int(rng.integers(0, 3)),
                span=float(SPAN if rng.random() < .5 else rng.uniform(3.4, 5.8)) if self.multi_speed else SPAN,
            )
            if self.direction_input and self.mixed_direction:
                fig8_chance = .25 if (self.multi_speed and speed_cap == MAX_SPEED) else (.20 if self.env.reverse else 0.0)
                if rng.random() < fig8_chance:
                    self.env.fixed_scenario = replace(self.env.fixed_scenario, figure8=True,
                                                      loop_width=rng.uniform(20., 26.),
                                                      loop_height=rng.uniform(10., 13.))
        observation, info = self.env.reset(seed=seed, options=options)
        self.leader_index = int(np.searchsorted(self.env.route.s, self.env.span))
        target_speed = (2.0 / 3.0) * self.env.speed_limit
        return self._observe(observation, [target_speed / self.env.speed_limit - CONDITIONED_CRUISE, 0.]), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        target_speed = (2.0 / 3.0) * self.env.speed_limit
        scale = 0.25 * (self.env.speed_limit / MAX_SPEED)
        leader_speed_error = (info['body_speed'][0] - target_speed) / scale
        reward -= self.speed_weight * leader_speed_error ** 2
        return self._observe(observation, action[:2]), reward, terminated, truncated, info

    def _observe(self, observation, leader_action):
        started = perf_counter_ns() if self.profile else 0
        preview, self.leader_index = rollout_features(self.env, leader_action, self.leader_index)
        value = np.r_[observation, preview]
        if self.direction_input:
            value = np.r_[value, -1. if self.env.reverse else 1.]
        result = value.astype(np.float32)
        # The demonstration teacher may read the unmasked vector; the actors never do.
        self.full_observation = result.copy()
        if self.observation_set == 'pose-only':
            slide_hat, slide_dot_hat, rel_sin = pose_derived_spacing(
                self.env.estimated_states, self.env.estimate_age, span=self.env.span)
            result[18] = np.float32(slide_hat / SLIDER_LIMIT)
            result[19] = np.float32(slide_dot_hat / 0.5)
            result[22] = np.float32(rel_sin)
        result[self.hidden] = 0.
        if self.profile:
            self.controller_preparation_ms = (perf_counter_ns() - started) * 1e-6
        return result


def teacher_action(observation, env=None):
    """Demonstration controller used for initialization only."""
    if env is not None and getattr(env, 'full_observation', None) is not None:
        observation = env.full_observation
    speed_limit = env.env.speed_limit if env is not None else MAX_SPEED
    cruise = (2.0 / 3.0) * speed_limit
    action = geometric_action(observation, cruise, speed_limit=speed_limit)
    action[0] = cruise / speed_limit - CONDITIONED_CRUISE
    slider = observation[18] * SLIDER_LIMIT
    slider_velocity = observation[19] * .5
    lead_speed = observation[2] * speed_limit
    follow_speed = .75 * lead_speed + .25 * cruise - 1.55 * slider - .45 * slider_velocity
    action[2] = follow_speed / speed_limit - CONDITIONED_CRUISE
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


def imitate(model, seed, *, mixed_direction=False, direction_input=False, observation='full', multi_speed=False):
    """Fit both actors on expert trajectories and on states the actors visit."""
    env = FieldPairEnv(mixed_direction=mixed_direction, direction_input=direction_input,
                       observation=observation, multi_speed=multi_speed)
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
    parser.add_argument('--observation', choices=sorted(HIDDEN_INPUTS), default='full',
                        help='inputs the actors may read; "pose-only" matches what the MPC receives')
    parser.add_argument('--multi-speed', action='store_true',
                        help='sample across 0.75-3.0 m/s caps and forward/reverse figure-eights')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.warm_from or args.adapt_from:
        env = FieldPairEnv(mixed_direction=args.mixed_direction, direction_input=args.direction_input,
                           observation=args.observation, multi_speed=args.multi_speed)
        try:
            if args.adapt_from and not args.direction_input:
                model = PPO.load(args.adapt_from, env=env, device='cpu', seed=args.seed)
            else:
                model = PPO(FieldPairPolicy, env, device='cpu', seed=args.seed, n_steps=256,
                        batch_size=256, n_epochs=4, gamma=.997, learning_rate=1.5e-5,
                        target_kl=.009, policy_kwargs={'ortho_init': False, 'log_std_init': -3.5})
                copy_role_weights(model, PPO.load(args.warm_from or args.adapt_from, device='cpu'))
            imitate(model, args.seed, mixed_direction=args.mixed_direction, direction_input=args.direction_input,
                    observation=args.observation, multi_speed=args.multi_speed)
            model.save(args.out / 'policy')
        finally:
            env.close()
        return

    env = SubprocVecEnv([lambda: Monitor(FieldPairEnv(slider_weight=args.slider_weight,
                                                    speed_weight=args.speed_weight,
                                                    mixed_direction=args.mixed_direction,
                                                    direction_input=args.direction_input,
                                                    observation=args.observation,
                                                    multi_speed=args.multi_speed),
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
