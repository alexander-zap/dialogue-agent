import json


class DialogConfig:
    def __init__(self, config_file_path=None):
        if config_file_path:
            with open(config_file_path, 'r') as f:
                config_json = json.load(f)
        else:
            config_json = None

        self.max_round_num = config_json['max_round_num'] if config_json else 0
        self.agent_request_slots = config_json['agent_request_slots'] if config_json else []
        self.agent_inform_slots = config_json['agent_inform_slots'] if config_json else []
        self.default_start_slot = config_json['default_start_slot'] if config_json else []
        self.no_query_slots = config_json['no_query_slots'] if config_json else []
        self.all_intents = config_json['all_intents'] if config_json else []
        self.agent_rule_requests = config_json['agent_rule_requests'] if config_json else []
        self.slot_name_translations = config_json['slot_name_translations'] if config_json else {}

        self.all_slots = sorted(list(set(self.agent_inform_slots + self.agent_request_slots)))

        # Possible actions for the agent
        self.feasible_agent_actions = [
            {'intent': 'done', 'inform_slots': {}, 'request_slots': []},  # Triggers closing of conversation
            {'intent': 'match_found', 'inform_slots': {}, 'request_slots': []}  # Signals a found match for a ticket
        ]

        # Add inform slots
        for slot in self.agent_inform_slots:
            if slot != 'ticket':
                self.feasible_agent_actions.append({'intent': 'inform',
                                                    'inform_slots': {slot: "PLACEHOLDER"},
                                                    'request_slots': []})

        # Add request slots
        for slot in self.agent_request_slots:
            self.feasible_agent_actions.append({'intent': 'request',
                                                'inform_slots': {},
                                                'request_slots': [slot]})


def init_config(config_file_path):
    global config
    config = DialogConfig(config_file_path)


config = DialogConfig()
