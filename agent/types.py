from typing import TypedDict, Dict, Optional

from agent.pretrained_policy import HardcodedAgent


class PretrainedAgent(TypedDict):
    prob: float  # Probability agent appears in training. Useless here
    eval: bool  # Whether or not to include in eval pools
    # Probability of using deterministic in training, defaults to True. Useless here
    p_deterministic_training: Optional[float]
    key: str  # The key to be used for the redis hash set, should be unique.


PretrainedAgents = Dict[HardcodedAgent, PretrainedAgent]
