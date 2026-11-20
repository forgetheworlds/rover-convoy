"""Independent learned leader and follower actors with a shared training critic."""

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from data import observation_names


def _network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, outputs))


class ConvoyActors(nn.Module):
    latent_dim_pi = 4
    latent_dim_vf = 64

    def __init__(self, preview_features=0, *, direction_input=False):
        super().__init__()
        names = observation_names('adaptive_speed')
        common = {'command_speed_limit', 'observation_delay'}
        follower_link = {'slider_position', 'slider_velocity', 'relative_grade',
                         'relative_heading_sin', 'leader_forward_velocity', 'leader_yaw_rate'}
        leader = [i for i, name in enumerate(names)
                  if name.startswith(('leader_', 'previous_leader_')) or name in common]
        follower = [i for i, name in enumerate(names)
                    if name.startswith(('follower_', 'previous_follower_')) or name in common | follower_link]
        follower.extend(range(len(names), len(names) + preview_features))
        if direction_input:
            leader.append(len(names) + preview_features - 1)
        self.register_buffer('leader_inputs', torch.tensor(leader))
        self.register_buffer('follower_inputs', torch.tensor(follower))
        self.leader = _network(len(leader), 2)
        self.follower = _network(len(follower), 2)
        self.critic = _network(len(names) + preview_features, self.latent_dim_vf)

    def forward_actor(self, observations):
        leader = self.leader(observations.index_select(-1, self.leader_inputs))
        follower = self.follower(observations.index_select(-1, self.follower_inputs))
        return torch.cat((leader, follower), dim=-1)

    def forward_critic(self, observations):
        return torch.tanh(self.critic(observations))

    def forward(self, observations):
        return self.forward_actor(observations), self.forward_critic(observations)


class LeaderFollowerPolicy(ActorCriticPolicy):
    """PPO uses joint rewards for training; each action head reads its own inputs."""

    def _build_mlp_extractor(self):
        if self.features_dim != 41:
            raise ValueError('Leader–follower PPO requires the 41-input environment.')
        self.mlp_extractor = ConvoyActors().to(self.device)

    def _build(self, lr_schedule):
        if self.use_sde:
            raise ValueError('Leader–follower PPO uses independent Gaussian exploration.')
        super()._build(lr_schedule)
        # The actor heads already output the two commands for each rover.
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
