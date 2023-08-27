import numpy as np
import itertools

from agent.discrete_policy import DiscretePolicy
from agent.types import PretrainedAgents
import os


class Matchmaker:

    def __init__(self, pretrained_agents: PretrainedAgents = None, full_team_evaluations=1, models_location = "./models"):
        """
        :param pretrained_agents: a configuration dict for how and how often to use pretrained agents in matchups.
        :param full_team_evaluations: The probability that a match uses all agents of the same type on a given team in evals.
        """
        if pretrained_agents is not None:
            self.consider_pretrained = True
            self.pretrained_agents_keys = [value.key for value in pretrained_agents.values() if value.eval == True]
        else:
            self.pretrained_agents_keys = []

        self.full_team_evaluations = full_team_evaluations
        models_location = os.path.abspath(models_location)
        self.models = os.listdir(models_location)

    def generate_matchup(self, n_blue, n_orange):
        full_team_match = np.random.random() < (self.full_team_evaluations)

        all_keys = self.pretrained_agents_keys + self.models

        if full_team_match:
            choice_idxs = np.random.choice(len(all_keys), size=2, replace=False)
            choice_deterministic =  np.random.choice(["deterministic", "stochastic"], size=2, replace=True)
            vs = [
                all_keys[choice_idxs[idx]]
                if all_keys[choice_idxs[idx]] in self.pretrained_agents_keys
                else f"{all_keys[choice_idxs[idx]]}-{choice_deterministic[idx]}"
                for idx in range(2)
            ]
            versions = [vs[0]] * n_blue
            versions += [vs[1]] * n_orange
        else:
            choice_idxs = np.random.choice(len(all_keys), size=n_blue + n_orange, replace=True)
            choice_deterministic =  np.random.choice(["deterministic", "stochastic"], size=n_blue + n_orange, replace=True)
            versions = [
                all_keys[choice_idxs[idx]]
                if all_keys[choice_idxs[idx]] in self.pretrained_agents_keys
                else f"{all_keys[choice_idxs[idx]]}-{choice_deterministic[idx]}"
                for idx in range(n_blue+n_orange)
            ]

        return versions
