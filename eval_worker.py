import copy

import sqlite3 as sql

import numpy as np
from redis import Redis
from rlgym_sim.envs import Match
import torch
from tabulate import tabulate


import agent.policy
from rlbot_agent import Agent
import rlgym_sim
from agent.discrete_policy import DiscretePolicy
from agent.types import PretrainedAgents
from generate_episode import generate_episode
from dynamic_gamemode_setter import DynamicGMSetter
from matchmaker.random_eval_matchmaker import Matchmaker


class EvalWorker:
    """
    Provides RedisRolloutGenerator with rollouts via a Redis server

     :param match: match object
     :param matchmaker: BaseMatchmaker object
     :param send_gamestates: Should gamestate data be sent back (increases data sent) - must send obs or gamestates
     :param send_obs: Should observations be send back (increases data sent) - must send obs or gamestates
     :param scoreboard: Scoreboard object
     :param pretrained_agents: PretrainedAgents typed dict
     :param live_progress: Show progress for eval matches
    """

    def __init__(self, match: Match, matchmaker: Matchmaker, output_file, tick_skip, initial_actors, send_gamestates=True,
                 send_obs=True, scoreboard=None, pretrained_agents: PretrainedAgents = None,
                 live_progress=True
                 ):

        self.matchmaker = matchmaker

        assert send_gamestates or send_obs, "Must have at least one of obs or states"

        self.pretrained_agents = {}
        self.pretrained_agents_keymap = {}
        if pretrained_agents is not None:
            self.pretrained_agents = pretrained_agents
            for agent, config in pretrained_agents.items():
                self.pretrained_agents_keymap[config["key"]] = agent

        self.send_gamestates = send_gamestates
        self.send_obs = send_obs
        self.gamemode_weights = {'1v1': 1 / 3, '2v2': 1 / 3, '3v3': 1 / 3}
        assert np.isclose(sum(self.gamemode_weights.values()),
                          1), "gamemode_weights must sum to 1"
        self.current_weights = copy.copy(self.gamemode_weights)

        self.scoreboard = scoreboard
        state_setter = DynamicGMSetter(match._state_setter)  # noqa Rangler made me do it
        self.set_team_size = state_setter.set_team_size
        match._state_setter = state_setter
        self.match = match
        self.env = rlgym_sim.gym.Gym(match=self.match, copy_gamestate_every_step=True, tick_skip=tick_skip, dodge_deadzone=0.5, gravity=1, boost_consumption=1)
        self.live_progress = live_progress
        self.fp = open(output_file, "a")
        self.initial_actors = initial_actors

    def select_gamemode(self, equal_likelihood):
        mode = np.random.choice(list(self.current_weights.keys()), p=list(
            self.current_weights.values()))
        if equal_likelihood:
            mode = np.random.choice(list(self.current_weights.keys()))
        b, o = mode.split("v")
        return int(b), int(o)

    @staticmethod
    def make_table(versions, ratings, blue, orange):
        version_info = []
        for v, r in zip(versions, ratings):
            if v == 'na':
                version_info.append(['Human', "N/A"])
            else:
                if isinstance(v, int):
                    v *= -1
                version_info.append([v, f"{r.mu:.2f}±{2 * r.sigma:.2f}"])

        blue_versions, blue_ratings = list(zip(*version_info[:blue]))
        orange_versions, orange_ratings = list(zip(*version_info[blue:])) if orange > 0 else list(((0,), ("N/A",)))

        if blue < orange:
            blue_versions += ("",) * (orange - blue)
            blue_ratings += ("",) * (orange - blue)
        elif orange < blue:
            orange_versions += ("",) * (blue - orange)
            orange_ratings += ("",) * (blue - orange)

        table_str = tabulate(list(zip(blue_versions, blue_ratings, orange_versions, orange_ratings)),
                             headers=["Blue", "rating", "Orange", "rating"], tablefmt="rounded_outline")

        return table_str

    def run(self):  # Mimics Thread
        """
        begin processing in already launched match and push to redis
        """
        n = 0
        while True:
            n += 1
            blue, orange = self.select_gamemode(equal_likelihood=True)
            versions = self.matchmaker.generate_matchup(blue, orange)
            agents = []
            for idx, version in enumerate(versions):
                # For any instances of HardcodedAgent, whose redis qualities keys are just the key in the keymap
                if version in self.pretrained_agents_keymap:
                    selected_agent = self.pretrained_agents_keymap[version]
                else:
                    file = "-".join(version.split("-")[:-1])
                    if version.endswith("deterministic"):
                        selected_agent = Agent(file, 1, self.initial_actors[idx])
                    elif version.endswith("stochastic"):
                        selected_agent = Agent(file, 0.5, self.initial_actors[idx])
                    else:
                        raise ValueError("Unknown version type")
                agents.append(selected_agent)

            self.set_team_size(blue, orange)

            result = generate_episode(self.env, agents, versions, progress=self.live_progress, scoreboard=self.scoreboard)
            self.fp.write(f"{versions[0]},{versions[-1]},{blue},{orange},{result['blue']},{result['orange']}\n")
            self.fp.flush()
            print(f"{versions[0]},{versions[-1]},{blue},{orange},{result['blue']},{result['orange']}")
            

