import copy
import json
import re
from collections import namedtuple

from dialogue_agent.action.useraction import UserAction
from dialogue_agent.gui.chat_application import ChatApplication
from dialogue_agent.dialog_config import max_round_num, all_slots, slot_name_translations
from dialogue_agent.util_functions import reward_function, agent_action_answered_user_request


class User(object):
    """" Class for interaction with real users """

    def __init__(self, nlu_path: str, gui: ChatApplication, use_voice=False):
        self.nlu_path = nlu_path
        self.gui = gui
        self.use_voice = use_voice
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
        self.constraint_check = False

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

        user_started_correctly = False
        while not user_started_correctly:
            nlu_response = self.ask_for_input()
            self.user_action.intent = nlu_response.intent
            if self.user_action.intent == "inform":
                self.add_inform_to_action(nlu_response.entities["slot_name"], nlu_response.entities["slot_value"])
                user_started_correctly = True
            elif self.user_action.intent == "request":
                self.request_slots.append(nlu_response.entities["slot_name"])
                user_started_correctly = True
            else:
                print("Please start with an inform or a request.")
        self.user_action.request_slots = copy.deepcopy(self.request_slots)

        done = False
        success = 0
        reward = reward_function(success)

        return self.user_action, reward, done, success

    def get_next_action(self, agent_action):
        """ Get next user action based on user input """

        agent_intent = agent_action.intent

        done = False
        success = 0

        # End dialogue immediately if turn maximum is reached or if agent is done
        if self.turn >= max_round_num:
            done = True
            success = -1
            self.user_action.intent = 'done'
        if agent_intent == "done":
            done = True
            self.user_action.intent = 'done'
            success = self.evaluate_success()

        agent_responsive = agent_action_answered_user_request(self.request_slots, agent_action)

        # Else parse user input and create UserAction
        if not done:
            while not self.user_action.intent:
                user_nlu_response = self.ask_for_input()
                if agent_intent == "inform" or agent_intent == "request":
                    self.process_normal_response(user_nlu_response)
                elif agent_intent == "match_found":
                    self.process_match_found_response(agent_action, user_nlu_response)

        # End dialogue if user replied that he is done
        if self.user_action.intent == 'done':
            done = True
            success = self.evaluate_success()

        reward = reward_function(success, agent_responsive)
        self.user_action.request_slots = copy.deepcopy(self.request_slots)
        self.request_slots.clear()
        return self.user_action, reward, done, success

    def process_normal_response(self, nlu_response):
        user_intent = nlu_response.intent
        entities = nlu_response.entities

        if user_intent == 'inform':
            self.user_action.intent = 'inform'
            self.add_inform_to_action(entities["slot_name"], entities["slot_value"])

        elif user_intent == 'request':
            self.user_action.intent = 'request'
            self.request_slots.append(entities["slot_name"])

        elif user_intent == 'thanks':
            self.user_action.intent = 'thanks'

        elif user_intent == 'done':
            self.user_action.intent = 'done'

    def process_match_found_response(self, agent_action, nlu_response):
        nlu_response_intent = nlu_response.intent

        # Agent needs to execute a 'match_found' action before a 'done' action to have a chance of "SUCCESS"
        self.constraint_check = True

        # 1) No match could be found with the user informs
        if agent_action.inform_slots["ticket"] == 'no match available':
            self.constraint_check = False
            print("No ticket could be found which matches your wishes.")

        # 2) User has to say yes to the ticket (all inform slots contained in agent action)
        if nlu_response_intent != 'yes':
            self.constraint_check = False

        if self.constraint_check:
            self.user_action.intent = 'accept'
        else:
            self.user_action.intent = 'reject'

    def ask_for_input(self):
        user_nlu_response = None
        while not user_nlu_response:
            if self.use_voice:
                user_utterance = self.gui.wait_for_speech_to_text().lower()
            else:
                user_utterance = self.gui.wait_for_user_message().lower()
            user_nlu_response = self.nlu_classify(user_utterance)
            if not user_nlu_response:
                print("I did not understand you. Please rephrase your answer.")
            else:
                user_entities = user_nlu_response[1]
                if 'slot_name' in user_entities and user_entities['slot_name'] not in all_slots:
                    print("I do not have any information about the slot '{}'. Please rephrase your answer."
                          .format(user_entities['slot_name']))
                    user_nlu_response = None
        return user_nlu_response

    def evaluate_success(self):
        return 1 if self.constraint_check else -1

    def add_inform_to_action(self, inform_slot, inform_value):
        self.user_action.inform_slots[inform_slot] = inform_value

    def nlu_classify(self, utterance):
        regex_entries = []
        with open(self.nlu_path, encoding="utf-8") as regex_nlu_file:
            regex_nlu = json.load(regex_nlu_file)
            for regex_entry in regex_nlu['patterns']:
                regex_entries.append((regex_entry['pattern'], regex_entry['intent'], regex_entry['entities']))

        nlu_response = namedtuple("NLU_Response", "intent entities")

        slot_translations = dict((k.lower(), v) for k, v in slot_name_translations.items())

        for pattern, intent, entity_keys in regex_entries:
            entity_dict = {}
            if re.match(pattern, utterance):
                match = re.search(pattern, utterance)
                groups = match.groups()
                for i in range(len(groups)):
                    entity_value = groups[i]
                    if "slot_name" in entity_keys and entity_value in slot_translations.keys():
                        entity_value = slot_translations[entity_value]
                    entity_dict.update({entity_keys[i]: entity_value})
                return nlu_response(intent, entity_dict)

        return None
