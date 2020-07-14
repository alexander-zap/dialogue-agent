from abc import ABC, abstractmethod
import random
from action.useraction import UserAction


class Usersim(ABC):

    """" Parent class for all user sims to inherit from """

    def __init__(self, goal_set):
        self.goal_set = goal_set
        self.goal = {}
        self.history_slots = {}
        self.request_slots = []
        self.rest_slots = {}
        self.constraint_check = False
        self.user_action = UserAction()
        self.turn = 0

        self.reset()

    def reset(self) -> None:
        """
        Reset user (new dialogue)
        :param self:
        'history_slots' : all slots which have been informed by user or agent so far
        'inform_slots' : inform slots that the user intents to inform in the current action being crafted
        'request_slots': request slots that the user wants to request in immediate future actions
        'rest_slots' : all inform and request slots from goal which have not been informed by user or agent
        """
        self.goal = self.sample_goal()
        # Add a "ticket" entry to the request slots of the goal (every user has the goal of requesting a ticket)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.history_slots = {}
        self.request_slots = []
        self.rest_slots = {}
        # Add the inform and request slots from the goal to the rest slots (slots to be informed/requested) of the user
        self.rest_slots.update(self.goal['inform_slots'])
        self.rest_slots.update(self.goal['request_slots'])
        self.constraint_check = False
        self.user_action = UserAction()
        self.turn = 0

    def sample_goal(self):
        # Randomly sample a user goal from the goal set
        sample_goal = random.choice(self.goal_set)
        return sample_goal

    def get_action(self, agent_action):
        if self.turn == 0:
            action = self.get_start_action()
        else:
            action = self.get_next_action(agent_action)
        self.turn += 1
        return action

    def add_inform_to_action(self, inform_slot, custom_inform_value=None):
        if custom_inform_value is not None:
            inform_value = custom_inform_value
        else:
            inform_value = self.goal['inform_slots'][inform_slot]
        self.user_action.inform_slots[inform_slot] = inform_value
        self.history_slots[inform_slot] = inform_value
        if inform_slot in self.rest_slots.keys():
            self.rest_slots.pop(inform_slot)

    @abstractmethod
    def get_start_action(self):
        self.user_action = UserAction()
        self.user_action.round_num = self.turn

    @abstractmethod
    def get_next_action(self, agent_action):
        self.user_action = UserAction()
        self.user_action.round_num = self.turn
