from dialog_config import slot_name_translations


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
        def translate(slot_name):
            """ Translates slot_name to German by slot_name_translations lookup."""
            reverted_slot_name_translations = {v: k for k, v in slot_name_translations.items()}
            if slot_name in reverted_slot_name_translations:
                slot_name = reverted_slot_name_translations[slot_name]
            return slot_name

        if self.intent == 'inform' or self.intent == 'match_found':
            inform_slot_key = translate(list(self.inform_slots.keys())[0])
            inform_slot_value = list(self.inform_slots.values())[0]
            if self.intent == 'inform':
                if inform_slot_value == 'no match available':
                    return "Ich konnte für {} leider keinen Match finden.".format(inform_slot_key)
                else:
                    return "Als {} wäre {} möglich.".format(inform_slot_key, inform_slot_value)
            elif self.intent == 'match_found':
                if 'no match available' in inform_slot_value:
                    return "Ich konnte leider kein passendes Ticket finden."
                else:
                    return "Kann ich Ihnen das Ticket {} empfehlen?".format(self.inform_slots)
        elif self.intent == 'request':
            request_slot_key = translate(self.request_slots[0])
            return "Was wünschen Sie als {}?".format(request_slot_key)
        elif self.intent == 'done':
            return "Auf Wiedersehen."
