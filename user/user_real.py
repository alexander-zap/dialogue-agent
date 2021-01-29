import copy
import json
import os
import re
from collections import namedtuple

from action.useraction import UserAction
from dialog_config import max_round_num
from util_functions import reward_function


class User(object):
    """" Class for interaction with real users """

    def __init__(self):
        self.turn = 0
        self.user_action = UserAction()
        self.request_slots = []
        self.constraint_check = False

    def reset(self) -> None:
        """
        Reset dialogue
        """
        self.turn = 0
        self.user_action = UserAction()
        self.request_slots = []

    def get_action(self, agent_action):
        self.user_action = UserAction()
        self.user_action.round_num = self.turn
        if self.turn == 0:
            action = self.get_start_action()
        else:
            action = self.get_next_action(agent_action)
        self.turn += 1
        return action

    def get_start_action(self):
        """ Get start user action based on user input """

        # TODO: Does real user need to start with a request?
        # TODO: Maybe it makes sense to inform first; something general, which is asked by the bot before (in turn "-1")
        #  e.g. for Saturn: "which category are you looking for?" - "category: hard drives"
        #  If implemented: Should we change this in the user simulator as well?

        self.user_action.request_slots = copy.deepcopy(self.request_slots)
        done = False
        success = 0
        reward = reward_function(success)
        return self.user_action, reward, done, success

    def get_next_action(self, agent_action):
        """ Get next user action based on user input """

        agent_intent = agent_action.intent

        success = 0
        done = False
        # End dialogue immediately if turn maximum is reached or if agent is done
        if self.turn >= max_round_num:
            done = True
            success = -1
            self.user_action.intent = 'done'
        elif agent_intent == "done":
            done = True
            self.user_action.intent = 'done'
            success = self.evaluate_success()
        else:
            user_nlu_response = self.ask_for_input()

            if agent_intent == "inform":
                self.parse_response_inform(agent_action, user_nlu_response)
            elif agent_intent == "request":
                self.parse_response_request(agent_action, user_nlu_response)
            elif agent_intent == "match_found":
                self.parse_response_match_found(agent_action, user_nlu_response)

        reward = reward_function(success)

        if self.user_action.intent in ['inform', 'reject', 'done']:
            # Inform, reject and done intents do not contain request slots
            self.request_slots.clear()
        self.user_action.request_slots = copy.deepcopy(self.request_slots)
        self.turn += 1
        return self.user_action, reward, done, success

    def parse_response_request(self, agent_action, nlu_response):
        agent_request_slot = agent_action.request_slots[0]
        intent = nlu_response.intent
        slot_value = nlu_response.slot_value

        # Case 1): Agent requests a slot which is in the goal inform slots -> Inform agent about requested slot
        # Case 2): Agent requests for slot that is in the goal request slots and it has been informed already
        # Case 4): Agent requests for slot that user does not care about (slot not in goal request or inform slots)
        if intent == 'inform' or intent == 'anything':
            self.user_action.intent = 'inform'
            slot_value = 'anything' if intent == 'anything' else slot_value
            self.add_inform_to_action(agent_request_slot, slot_value)

        # Case 3): Agent requests for slot that is in the goal request slots and it has not been informed yet
        elif intent == 'request':
            self.user_action.intent = 'request'
            self.request_slots.clear()
            self.request_slots.append(slot_value)

    def parse_response_inform(self, agent_action, nlu_response):
        agent_inform_slot = list(agent_action.inform_slots.keys())[0]

        intent = nlu_response.intent
        slot_value = nlu_response.slot_value

        if agent_inform_slot in self.request_slots:
            self.request_slots.remove(agent_inform_slot)

        # Case 1): Correct agent if value of agent inform slot does match user goal
        # Case 2): Follow up: pick some slot to inform
        if intent == 'inform':
            self.user_action.intent = 'inform'
            self.add_inform_to_action(agent_inform_slot, slot_value)

        # Case 2): Follow up: pick some slot to request
        elif intent == 'request':
            self.user_action.intent = 'request'
            self.request_slots.append(slot_value)

        elif intent == 'ticket':
            self.user_action.intent = 'request'
            self.request_slots.append("ticket")

        # Case 2.c): Nothing to say, respond with thanks
        else:
            self.user_action.intent = 'thanks'

    def parse_response_match_found(self, agent_action, nlu_response):
        intent = nlu_response.intent

        # Agent needs to execute a 'match_found' action before a 'done' action to have a chance of "SUCCESS"
        self.constraint_check = "SUCCESS"

        # 1) No match could be found with the user informs
        if agent_action.inform_slots["ticket"] == 'no match available':
            self.constraint_check = "FAIL"
            print("No ticket could be found which matches your wishes.")

        # 2) User has to say yes to the ticket (all inform slots contained in agent action)
        if intent == 'yes':
            pass
        else:
            self.constraint_check = "FAIL"

        if self.constraint_check == "FAIL":
            self.user_action.intent = 'reject'
        else:
            self.user_action.intent = 'thanks'

    def ask_for_input(self):
        user_nlu_response = None
        while not user_nlu_response:
            user_utterance = input(">>>").lower()
            user_nlu_response = self.nlu_classify(user_utterance)
            if not user_nlu_response:
                print("I did not understand you. Please rephrase your answer.")
        return user_nlu_response

    def evaluate_success(self):
        if self.constraint_check == "FAIL":
            return -1
        return 1

    def add_inform_to_action(self, inform_slot, inform_value):
        self.user_action.inform_slots[inform_slot] = inform_value

    @staticmethod
    def nlu_classify(utterance):
        location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        nlu_path = os.path.join(location, 'regex_nlu.json')
        pattern_intent_tuples = []
        with open(nlu_path) as regex_json:
            patterns_json = json.load(regex_json)
            for regex_entry in patterns_json['patterns']:
                pattern_intent_tuples.append((regex_entry['pattern'], regex_entry['intent']))

        nlu_response = namedtuple("NLU_Response", "intent slot_value")

        for pattern, intent in pattern_intent_tuples:
            if re.match(pattern, utterance):
                slot_value = re.findall(pattern, utterance)[0]
                return nlu_response(intent, slot_value)

        return None
