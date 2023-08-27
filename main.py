from matchmaker.random_eval_matchmaker import Matchmaker
from necto_obs import NectoObsBuilder
from eval_worker import EvalWorker
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward
from torch import set_num_threads
import dotenv
from rlgym_sim.envs import Match as Sim_Match
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
    NoTouchTimeoutCondition
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.action_parsers import DefaultAction
import sys
import torch.jit

set_num_threads(1)

if __name__ == "__main__":
    dotenv.load_dotenv()
    frame_skip = 8
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    streamer_mode = False
    local = True
    auto_minimize = True
    
    team_size = 3
    dynamic_game = True

    match = Sim_Match(
        spawn_opponents=True,
        team_size=team_size,
        state_setter=DefaultState(),
        obs_builder=NectoObsBuilder(),
        terminal_conditions=[GoalScoredCondition(),
                             NoTouchTimeoutCondition(fps * 15),
                             TimeoutCondition(fps * 300),
                             ],
        reward_function=ConstantReward(),
        action_parser=DefaultAction()
    )
    pretrained_agents = None

    matchmaker = Matchmaker(pretrained_agents=pretrained_agents)

    initial_actors = []
    for idx in range(6):
        initial_actors.append(torch.jit.load("necto_rlbot_checkpoint.pt"))

    worker = EvalWorker(match,
                        matchmaker=matchmaker, output_file=sys.argv[1],
                        pretrained_agents=pretrained_agents,
                        tick_skip=frame_skip, live_progress=False
                        )

    worker.env._match._obs_builder.env = worker.env  # noqa
    worker.run()
