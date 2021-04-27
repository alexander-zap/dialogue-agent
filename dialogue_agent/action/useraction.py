class UserAction:

    def __init__(self):
        self.intent = ''
        self.inform_slots = {}
        self.request_slots = []
        self.round_num = 0
        self.speaker = "User"

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)
