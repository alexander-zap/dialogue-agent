import copy

from dialogue_agent.action import AgentAction
from dialogue_agent.dialog_config import feasible_agent_actions, max_round_num


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


def agent_action_answered_user_request(user_request_slots=None, agent_action=None):
    """
    Check whether user_request was answered in agent_informs

    :param user_request_slots: Slots user requested in previous user_action
    :param agent_action: Action of agent as response to previous user_action
    :return: Boolean, whether a request slot from previous user_action was contained in the inform slots of agent_action
    """
    agent_inform_slots = set(agent_action.inform_slots.keys())
    return bool(set(user_request_slots).intersection(set(agent_inform_slots)))


def reward_function(success, agent_responsive=None):
    reward = -1
    if success == 1:
        reward += 2 * max_round_num
    elif success == -1:
        reward -= max_round_num
    elif agent_responsive is not None:
        if agent_responsive:
            reward += 8
        else:
            reward -= 6
    return reward
