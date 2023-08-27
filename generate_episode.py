from typing import List

import numpy as np
import torch
from rlgym_sim.gym import Gym
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition
from rlgym_tools.extra_terminals.game_condition import GameCondition
from scoreboard import Scoreboard
from agent.pretrained_policy import HardcodedAgent
from tqdm import tqdm


def generate_episode(env: Gym, policies, versions, eval_setter=DefaultState(),
                     progress=False, scoreboard: Scoreboard = None
                     ):
    """
    create experience buffer data by interacting with the environment(s)
    """
    if progress:
        progress = tqdm(unit=" steps")
    else:
        progress = None

    terminals = env._match._terminal_conditions  # noqa
    reward = env._match._reward_fn  # noqa
    # game_condition = GameCondition(tick_skip=env._game.tick_skip)  # noqa
    env._match._terminal_conditions = [GoalScoredCondition()]  # noqa
    state_setter = env._match._state_setter.setter  # noqa
    env._match._state_setter.setter = eval_setter  # noqa

    if scoreboard is not None:
        random_resets = scoreboard.random_resets
        scoreboard.random_resets = False
    observations, info = env.reset(return_info=True)
    result = 0

    last_state = info['state']  # game_state for obs_building of other agents
    pretrained_idxs = [idx for idx, v in enumerate(
        versions) if isinstance(policies[idx], HardcodedAgent)]
        
    b = o = 0
    with torch.no_grad():
        tick = [0] * len(policies)
        do_selector = [True] * len(policies)
        last_actions = [None] * len(policies)
        first_step = True
        while True:
            # all_indices = []
            all_actions = []
            # all_log_probs = []
            # all_actions = [None] * len(policies)

            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]

            index = 0
            for policy, obs in zip(policies, observations):
                if isinstance(policy, HardcodedAgent):
                    actions = policy.act(last_state, index)

                    # make sure output is in correct format
                    if not isinstance(observations, np.ndarray):
                        actions = np.array(actions)

                    # TODO: add converter that takes normal 8 actions into action space
                    # actions = env._match._action_parser.convert_to_action_space(actions)

                    all_actions.append(actions)

                else:
                    all_actions.append(policy.act(obs))

                index += 1

            # to allow different action spaces, pad out short ones to longest length (assume later unpadding in parser)
            # length = max([a.shape[0] for a in all_actions])
            # padded_actions = []
            # for a in all_actions:
            #     action = np.pad(
            #         a.astype('float64'), (0, length - a.size), 'constant', constant_values=np.NAN)
            #     padded_actions.append(action)

            # all_actions = padded_actions
            # TEST OUT ABOVE TO DEAL WITH VARIABLE LENGTH

            all_actions = np.vstack(all_actions)
            old_obs = observations
            observations, rewards, done, info = env.step(all_actions)

            if progress is not None:
                progress.update()
                igt = progress.n * env._game.tick_skip / 120  # noqa
                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                prog_str += f", BLUE {b} - {o} ORANGE"
                progress.set_postfix_str(prog_str)

            if done:
                result += info["result"]
                if info["result"] > 0:
                    b += 1
                elif info["result"] < 0:
                    o += 1
                break

            last_state = info['state']

    if scoreboard is not None:
        scoreboard.random_resets = random_resets  # noqa Checked above

    env._match._state_setter.setter = state_setter
    if progress is not None:
        progress.close()

    return {"blue": b, "orange": o}
