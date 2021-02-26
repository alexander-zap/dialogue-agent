import copy

from action.agentaction import AgentAction
from dialog_config import feasible_agent_actions, max_round_num


def index_to_agent_action(index):
    agent_action_raw = copy.deepcopy(feasible_agent_actions[index])
    agent_action = AgentAction()
    agent_action.intent = agent_action_raw['intent']
    agent_action.inform_slots = agent_action_raw['inform_slots']
    agent_action.request_slots = agent_action_raw['request_slots']
    agent_action.feasible_action_index = index
    return agent_action


def raw_agent_action_to_index(raw_agent_action):
    feasible_agent_action_index = feasible_agent_actions.index(raw_agent_action)
    return feasible_agent_action_index


def reward_function(success, user_request_slots=None, prev_agent_action: AgentAction = None):
    def user_request_in_agent_inform():
        agent_inform_slots = set(prev_agent_action.inform_slots.keys())
        return bool(set(user_request_slots).intersection(set(agent_inform_slots)))

    reward = -1
    if success == 1:
        reward += 2 * max_round_num
    elif success == -1:
        reward -= max_round_num
    elif user_request_slots and prev_agent_action:
        if user_request_in_agent_inform():
            reward -= 5
    return reward
