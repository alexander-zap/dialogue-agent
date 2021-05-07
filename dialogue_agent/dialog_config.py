import json
import os

from dialogue_agent.locations import RESOURCES_PATH

max_round_num = 20

agent_request_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating',
                       'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor',
                       'description', 'other', 'numberofkids', 'numberofpeople']

agent_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating',
                      'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor',
                      'description', 'other', 'numberofkids', 'ticket']

# Possible actions for the agent
feasible_agent_actions = [
    {'intent': 'done', 'inform_slots': {}, 'request_slots': []},  # Triggers closing of conversation
    {'intent': 'match_found', 'inform_slots': {}, 'request_slots': []}  # Signals a found match for a ticket
]

# Add inform slots
for slot in agent_inform_slots:
    if slot != 'ticket':
        feasible_agent_actions.append({'intent': 'inform',
                                       'inform_slots': {slot: "PLACEHOLDER"},
                                       'request_slots': []})

# Add request slots
for slot in agent_request_slots:
    feasible_agent_actions.append({'intent': 'request',
                                   'inform_slots': {},
                                   'request_slots': [slot]})

# These are possible inform slot keys that cannot be used to check current_inform_slots
no_query_slots = ['numberofpeople', 'ticket']

# Action order for agents warm-up phase (must be part of feasible_agent_actions)
agent_rule_requests = [{'intent': 'request', 'inform_slots': {}, 'request_slots': ['moviename']},
                       {'intent': 'request', 'inform_slots': {}, 'request_slots': ['starttime']},
                       {'intent': 'request', 'inform_slots': {}, 'request_slots': ['city']},
                       {'intent': 'request', 'inform_slots': {}, 'request_slots': ['date']},
                       {'intent': 'request', 'inform_slots': {}, 'request_slots': ['theater']},
                       {'intent': 'request', 'inform_slots': {}, 'request_slots': ['numberofpeople']},
                       {'intent': 'match_found', 'inform_slots': {}, 'request_slots': []},
                       {'intent': 'done', 'inform_slots': {}, 'request_slots': []}]

# All possible intents (for one-hot conversion in state representation)
all_intents = ['inform', 'request', 'thanks', 'match_found', 'accept', 'reject', 'done']

# All possible slots (for one-hot conversion in state representation)
all_slots = sorted(list(set(agent_inform_slots + agent_request_slots)))

# Dictionary containing translations from German to English slot_names
slot_name_translations = json.load(
    open(
        os.path.join(os.path.dirname(RESOURCES_PATH), "slot_name_translations.json"),
        "r", encoding="utf-8"
    )
)
