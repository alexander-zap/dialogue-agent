class AgentAction:

    def __init__(self):
        self.intent = ''
        self.inform_slots = {}
        self.request_slots = []
        self.feasible_action_index = 0
        self.round_num = 0
        self.speaker = "Agent"

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)

    def to_utterance(self):
        if self.intent == 'inform' or self.intent == 'match_found':
            inform_key = list(self.inform_slots.keys())[0]
            inform_value = list(self.inform_slots.values())[0]
            if self.intent == 'inform':
                if inform_value == 'no_match_available':
                    return "Ich konnte für {} leider keinen Match finden.".format(inform_key)
                else:
                    return "Als {} wäre {} möglich.".format(inform_key, inform_value)
            elif self.intent == 'match_found':
                if 'no_match_available' in inform_value:
                    return "Ich konnte leider kein passendes Ticket finden."
                else:
                    return "Kann ich Ihnen das Ticket {} empfehlen?".format(self.inform_slots)
        elif self.intent == 'request':
            return "Was wünschen Sie als {}?".format(self.request_slots[0])
        elif self.intent == 'done':
            return "Auf Wiedersehen."
