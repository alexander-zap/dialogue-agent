import random

from agent import Agent
from dialogue_agent.util_functions import index_to_agent_action


class RandomAgent(Agent):
    def build_model(self):
        pass

    # Chooses random action
    def get_greedy_action(self, obs):
        action_index = random.randrange(self.n_actions)
        return index_to_agent_action(action_index)

    def replay(self):
        pass
