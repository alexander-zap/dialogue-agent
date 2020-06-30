from agent.agent import Agent
from util_functions import index_to_agent_action
import random


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, **kwargs):
        pass

    # Chooses random action
    def choose_action(self, obs, warm_up):
        action_index = random.randrange(self.n_actions)
        return index_to_agent_action(action_index)
