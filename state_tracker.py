import copy
import random
import numpy as np
from dialog_config import all_intents, all_slots, max_round_num
from action.useraction import UserAction
from action.agentaction import AgentAction
from db_query import DBQuery


class StateTracker:
    def __init__(self, database):
        self.db_helper = DBQuery(database)
        self.current_informs = {}
        self.history = []
        self.round_num = 0

    def reset(self):
        self.current_informs = {}
        self.history = []
        self.round_num = 0

    # Update history and current informs of the state tracker with the input user action
    def update_state_agent(self, agent_action: AgentAction):
        # Fill the informs lots of the action by querying the database
        agent_action = self.fill_agent_action(agent_action)
        if agent_action.intent == 'inform':
            for key, value in agent_action.inform_slots.items():
                # Update the current informs with the chosen agent inform slots
                self.current_informs[key] = value
        elif agent_action.intent == 'match_found':
            # Update the value of ticket in current informs with the new value found in the database
            # Do not update other current_informs with agent_action inform slots, since they are from the full ticket
            self.current_informs['ticket'] = agent_action.inform_slots['ticket']
        agent_action.round_num = self.round_num
        # Add action to history
        self.history.append(agent_action)
        self.round_num += 1

    def fill_agent_action(self, agent_action):
        if agent_action.intent == 'inform':
            for slot in agent_action.inform_slots:
                # Fill the inform slot of the action by querying the database with the current informs as constraints
                agent_action.inform_slots[slot] = self.db_helper.get_best_slot_value(slot, self.current_informs)
        elif agent_action.intent == 'match_found':
            # Get all tickets from the database which match the current informs
            db_results = self.db_helper.get_matching_db_entries(self.current_informs)
            # If there are matching tickets then set the inform slots of the agent action to the slots of a ticket
            # from this list and set the value of the ticket-slot to the ID of this ticket
            if db_results:
                # Pick random entry from the dict
                db_id, inform_slots = random.choice(list(db_results.items()))
                agent_action.inform_slots = copy.deepcopy(inform_slots)
                agent_action.inform_slots['ticket'] = str(db_id)
            # Else set the value of the ticket-slot to ‘no match available’
            else:
                agent_action.inform_slots['ticket'] = 'no match available'
        return agent_action

    # Update history and current informs of the state tracker with the input user action
    def update_state_user(self, user_action: UserAction):
        for key, value in user_action.inform_slots.items():
            self.current_informs[key] = value
        self.history.append(user_action)

    def get_state(self, done=False):
        # If done then fill state with zeros
        if done:
            return np.zeros(self.state_size())

        last_user_action = self.history[-1] if len(self.history) > 0 else None
        last_agent_action = self.history[-2] if len(self.history) > 1 else None

        # Get database info that is useful for the agent
        db_results_dict = self.db_helper.count_matches_per_slot(self.current_informs)

        num_intents = len(all_intents)
        num_slots = len(all_slots)

        # Create one-hot of intents to represent the current user action
        user_act_rep = np.zeros((num_intents,))
        if last_user_action:
            user_act_rep[all_intents.index(last_user_action.intent)] = 1.0

        # Create bag of inform slots representation to represent the current user action
        user_inform_slots_rep = np.zeros((num_slots,))
        if last_user_action:
            for key in last_user_action.inform_slots.keys():
                user_inform_slots_rep[all_slots.index(key)] = 1.0

        # Create bag of request slots representation to represent the current user action
        user_request_slots_rep = np.zeros((num_slots,))
        if last_user_action:
            for slot in last_user_action.request_slots:
                user_request_slots_rep[all_slots.index(slot)] = 1.0

        # Create bag of current informs representation to represent all current informs
        current_slots_rep = np.zeros((num_slots,))
        for key in self.current_informs:
            current_slots_rep[all_slots.index(key)] = 1.0

        # Encode last agent intent
        agent_act_rep = np.zeros((num_intents,))
        if last_agent_action:
            agent_act_rep[all_intents.index(last_agent_action.intent)] = 1.0

        # Encode last agent inform slots
        agent_inform_slots_rep = np.zeros((num_slots,))
        if last_agent_action:
            for key in last_agent_action.inform_slots.keys():
                agent_inform_slots_rep[all_slots.index(key)] = 1.0

        # Encode last agent request slots
        agent_request_slots_rep = np.zeros((num_slots,))
        if last_agent_action:
            for slot in last_agent_action.request_slots:
                agent_request_slots_rep[all_slots.index(slot)] = 1.0

        # Value representation of the round num
        turn_rep = np.zeros((1,)) + self.round_num / 5.

        # One-hot representation of the round num
        turn_one_hot_rep = np.zeros((max_round_num,))
        turn_one_hot_rep[self.round_num] = 1.0

        # Representation of DB query results (scaled counts)
        kb_count_rep = np.zeros((num_slots + 1,)) + db_results_dict['all_slots'] / 100.
        for key in db_results_dict.keys():
            if key in all_slots:
                kb_count_rep[all_slots.index(key)] = db_results_dict[key] / 100.

        # Representation of DB query results (binary)
        kb_binary_rep = np.zeros((num_slots + 1,)) + np.sum(db_results_dict['all_slots'] > 0.)
        for key in db_results_dict.keys():
            if key in all_slots:
                kb_binary_rep[all_slots.index(key)] = np.sum(db_results_dict[key] > 0.)

        state_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_one_hot_rep, kb_binary_rep,
             kb_count_rep])

        return state_representation

    @staticmethod
    def state_size():
        return 2 * len(all_intents) + 7 * len(all_slots) + 3 + max_round_num
