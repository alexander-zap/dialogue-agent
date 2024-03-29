import copy
import random

import dialogue_agent.dialog_config as dia_config
from dialogue_agent.util_functions import reward_function, agent_action_answered_user_request
from .usersim import Usersim


class RulebasedUsersim(Usersim):
    def __init__(self, goal_set):
        super().__init__(goal_set)

    def reset(self) -> None:
        super().reset()

    def sample_goal(self):
        return super().sample_goal()

    def get_action(self, agent_action):
        return super().get_action(agent_action)

    def get_start_action(self):
        """ Randomly samples a start action based on user goal """

        # User simulator always tries to start with a request
        self.user_action.intent = "request"

        # Define request slot of start action by sampling one slot from the goal request slots
        request_slot_set = list(self.goal['request_slots'].keys())
        # Start action request slot should not be "ticket"
        if "ticket" in request_slot_set:
            request_slot_set.remove("ticket")
        # If other request slots are left in the goal, then sample one randomly
        if request_slot_set:
            start_request_slot = random.choice(request_slot_set)
            self.request_slots.append(start_request_slot)
        # If no other request slots are left in the goal, then no request slot is sampled and this is an inform action
        else:
            self.user_action.intent = 'inform'

        # Add inform slot to start action by sampling one slot from the goal inform slots
        if self.goal['inform_slots']:
            start_inform_slot = random.choice(list(self.goal['inform_slots'].keys()))
            self.add_inform_to_action(start_inform_slot)

        self.user_action.request_slots = copy.deepcopy(self.request_slots)
        done = False
        success = 0
        reward = reward_function(success)
        return self.user_action, reward, done, success

    def get_next_action(self, agent_action):
        """ Generate next user action based on last agent action and user state """

        agent_intent = agent_action.intent

        done = False
        success = 0

        # End episode if turn maximum is reached or if agent is done
        if self.turn >= dia_config.config.max_round_num:
            done = True
            success = -1
            self.user_action.intent = 'done'
        if agent_intent == "done":
            done = True
            self.user_action.intent = 'done'
            success = self.evaluate_success()

        agent_responsive = agent_action_answered_user_request(self.request_slots, agent_action)

        if agent_intent == "inform":
            self.response_inform(agent_action)
        elif agent_intent == "request":
            self.response_request(agent_action)
        elif agent_intent == "match_found":
            self.response_match_found(agent_action)

        if self.user_action.intent in ['inform', 'reject', 'done']:
            # Inform, reject and done intents do not contain request slots
            self.request_slots.clear()

        reward = reward_function(success, agent_responsive)
        self.user_action.request_slots = copy.deepcopy(self.request_slots)
        return self.user_action, reward, done, success

    def response_request(self, agent_action):
        agent_request_slot = agent_action.request_slots[0]

        # Case 1): Agent requests a slot which is in the goal inform slots -> Inform agent about requested slot
        if agent_request_slot in self.goal['inform_slots']:
            # User intent will be inform
            self.user_action.intent = 'inform'
            # Add requested slot to inform slot
            self.add_inform_to_action(agent_request_slot)

        # Case 2): Agent requests for slot that is in the goal request slots and it has been informed already
        elif agent_request_slot in self.goal['request_slots'] and agent_request_slot in self.history_slots.keys():
            self.user_action.intent = 'inform'
            # Repeat that has been said before (from history slots)
            self.add_inform_to_action(agent_request_slot, custom_inform_value=self.history_slots[agent_request_slot])

        # Case 3): Agent requests for slot that is in the goal request slots and it has not been informed yet
        elif agent_request_slot in self.goal['request_slots'] and agent_request_slot in self.rest_slots.keys():
            self.user_action.intent = 'request'
            self.request_slots.clear()
            self.request_slots.append(agent_request_slot)
            # Request for the slot and additionally (if possible) add an inform slot from rest slots
            rest_informs = {}
            for key, value in list(self.rest_slots.items()):
                if value != 'UNK':
                    rest_informs[key] = value
            if rest_informs:
                key_choice, value_choice = random.choice(list(rest_informs.items()))
                # Inform agent about additional inform slot
                self.add_inform_to_action(key_choice, custom_inform_value=value_choice)

        # Case 4): Agent requests for slot that user does not care about (slot not in goal request or inform slots)
        else:
            self.user_action.intent = 'inform'
            # Reply 'anything' as value for requested inform slot
            self.add_inform_to_action(agent_request_slot, custom_inform_value='anything')

    def response_inform(self, agent_action):
        agent_inform_slot = list(agent_action.inform_slots.keys())[0]
        agent_inform_value = agent_action.inform_slots[agent_inform_slot]
        # Add informed slot by agent to history_slots, remove slot from rest slots and request slots of the user
        self.history_slots[agent_inform_slot] = agent_inform_value
        self.rest_slots.pop(agent_inform_slot, None)
        if agent_inform_slot in self.request_slots:
            self.request_slots.remove(agent_inform_slot)

        # Case 1): Correct agent if value of agent inform slot does match user goal
        if agent_inform_slot in self.goal['inform_slots'].keys() \
                and agent_inform_value != self.goal['inform_slots'][agent_inform_slot]:
            self.user_action.intent = 'inform'
            # return inform slot, add to history slots and remove from rest slots (last step not needed)
            self.add_inform_to_action(agent_inform_slot)

        # Case 2): Follow up: pick some slot to request or inform
        else:

            # Case 2.a): Request something from already existing request slots if available
            if self.request_slots:
                self.user_action.intent = 'request'

            # Case 2.b): If something to say in rest slots, pick something
            elif self.rest_slots:
                # Here the ticket is being removed from rest (if it is in there) so that it selects another slot,
                # whether its a request or inform, but if ticket is the only one in rest then just request it
                ticket_value = self.rest_slots.pop("ticket", None)
                if self.rest_slots:
                    slot_key, slot_value = random.choice(list(self.rest_slots.items()))
                    # Inform from rest slots
                    if slot_value != 'UNK':
                        self.user_action.intent = 'inform'
                        self.add_inform_to_action(slot_key, custom_inform_value=slot_value)
                    # Request from rest slots
                    else:
                        self.user_action.intent = 'request'
                        self.request_slots.append(slot_key)
                # Nothing left ot request but ticket, so request ticket
                else:
                    self.user_action.intent = 'request'
                    self.request_slots.append("ticket")
                # Add ticket back to rest slots if it was there
                if ticket_value is not None:
                    self.rest_slots["ticket"] = 'UNK'

            # Case 2.c): Nothing to say, respond with thanks
            else:
                self.user_action.intent = 'thanks'

    def response_match_found(self, agent_action):
        # Agent needs to execute a 'match_found' action before a 'done' action to have a chance of "SUCCESS"
        self.constraint_check = True

        # Add the ticket slot to history slots and remove from rest and request slots
        self.history_slots["ticket"] = str(agent_action.inform_slots["ticket"])
        self.rest_slots.pop("ticket", None)
        if "ticket" in self.request_slots:
            self.request_slots.remove("ticket")

        # 1) No match could be found with the user informs
        if agent_action.inform_slots["ticket"] == 'no match available':
            self.constraint_check = False

        # 2) All goal inform slots must be in agent action and all slot values must be the same
        for slot, value in self.goal['inform_slots'].items():
            # No query indicates slots that cannot be in a ticket so they should not be checked
            if slot in dia_config.config.no_query_slots:
                continue
            if value != agent_action.inform_slots.get(slot, None):
                self.constraint_check = False
                break

        if not self.constraint_check:
            self.user_action.intent = 'reject'
        else:
            self.user_action.intent = 'accept'

    def evaluate_success(self):

        if not self.constraint_check:
            return -1

        # Rest slots must be empty for successful interaction (no goal inform or request slots left)
        if self.rest_slots:
            return -1

        return 1
