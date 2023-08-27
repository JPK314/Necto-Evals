from typing import Optional, List, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from agent.policy import Policy


class DiscretePolicy(Policy):
    def __init__(self, net: nn.Module, shape: Tuple[int, ...] = (3,) * 5 + (2,) * 3, deterministic=False):
        super().__init__(deterministic)
        self.net = net
        self.shape = shape

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def get_action_distribution(self, obs, initial_choice_block_indices=None):
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        elif isinstance(obs, tuple):
            obs = tuple(o if isinstance(o, th.Tensor) else th.from_numpy(o).float() for o in obs)

        logits = self(obs)

        if isinstance(logits, th.Tensor):
            logits = (logits,)

        if initial_choice_block_indices is not None:
            for i in range(len(logits[0])):
                for j in initial_choice_block_indices:
                    logits[0][i][j] = -th.inf

        max_shape = max(self.shape)
        logits = th.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in logits
            ],
            dim=1
        )

        return Categorical(logits=logits)

    def sample_action(
            self,
            distribution: Categorical,
            deterministic=None
    ):
        if deterministic is None:
            deterministic = self.deterministic
        if deterministic:
            action_indices = th.argmax(distribution.logits, dim=-1)
        else:
            action_indices = distribution.sample()

        return action_indices

    def log_prob(self, distribution: Categorical, selected_action):
        log_prob = distribution.log_prob(selected_action).sum(dim=-1)
        return log_prob

    def entropy(self, distribution: Categorical, selected_action):
        entropy = distribution.entropy().sum(dim=-1)
        return entropy

    def env_compatible(self, action):
        if isinstance(action, th.Tensor):
            action = action.numpy()
        return action
