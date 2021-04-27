import copy
from collections import defaultdict

from dialog_config import no_query_slots


class DBQuery:
    """Queries the database for the state tracker. Slightly modified version of https://github.com/maxbren/GO-Bot-DRL"""

    def __init__(self, database):
        """
        The constructor for DBQuery.

        Parameters:
            database (dict): The database in the format dict(long: dict)
        """

        self.database = database
        # {frozenset: {string: int}} A dict of dicts
        self.cached_db_slot = defaultdict(dict)
        # {frozenset: {'#': {'slot': 'value'}}} A dict of dicts of dicts, a dict of DB sub-dicts
        self.cached_db = defaultdict(dict)
        self.no_query = no_query_slots
        self.match_key = 'ticket'

    def get_matching_db_entries(self, current_inform_slots):
        """
        Get all items in the database that fit the current constraints.

        Looks at each item in the database and if its slots contain all constraints and their values match then the item
        is added to the return dict.

        Parameters:
            current_inform_slots (dict): The current informs

        Returns:
            dict: The available items in the database
        """
        # Filter non-queryable items and keys with the value 'anything' since those are inconsequential to the
        # current_inform_slots
        new_constraints = {k: v for k, v in current_inform_slots.items()
                           if k not in self.no_query and v is not 'anything'}

        inform_items = frozenset(new_constraints.items())
        cache_return = self.cached_db[inform_items]
        if cache_return:
            return cache_return

        available_options = {}
        for db_id in self.database.keys():
            current_option_dict = self.database[db_id]
            match = True
            # Check all the constraint values against the db values and if there is a mismatch don't store
            for k, v in new_constraints.items():
                if k not in current_option_dict.keys():
                    match = False
                    break
                if str(v).lower() != str(current_option_dict[k]).lower():
                    match = False
                    break
            if match:
                available_options.update({db_id: current_option_dict})
                # Update cache
                self.cached_db[inform_items].update({db_id: current_option_dict})

        # Update cache
        self.cached_db[inform_items] = available_options

        return available_options

    def get_best_slot_value(self, inform_slot, current_inform_slots):
        """
        Given the current informs fill the inform that need to be filled with values from the database

        Searches through the database to fill the inform slots with PLACEHOLDER with values that work given the current
        current_inform_slots of the current episode.

        Parameters:
            inform_slot (str): Inform slot to be filled with a value
            current_inform_slots (dict): Current inform slots with values from the StateTracker

        Returns:
            dict: best inform_slot_value for the given inform_slot_to_fill
        """

        # This removes the inform we want to fill from the current informs if it is present in the current informs
        # so it can be re-queried
        current_informs = copy.deepcopy(current_inform_slots)
        current_informs.pop(inform_slot, None)

        # db_results is a dict of dict in the same exact format as the db, it is just a subset of the db
        db_results = self.get_matching_db_entries(current_informs)

        value_distribution = self._count_slot_value_distribution(inform_slot, db_results)
        if value_distribution:
            # Get slot with max value (ie slot value with highest count of available results)
            return max(value_distribution, key=value_distribution.get)
        else:
            return 'no match available'

    @staticmethod
    def _count_slot_value_distribution(slot, db_subdict):
        """
        Return a dict of the different values and occurrences of each, given a key, from a sub-dict of database

        Parameters:
            slot (string): The slot to be counted
            db_subdict (dict): A sub-dict of the database

        Returns:
            dict: The values and their occurrences given the key
        """

        slot_values = defaultdict(int)  # init to 0
        for db_id in db_subdict.keys():
            current_option_dict = db_subdict[db_id]
            # If there is a match
            if slot in current_option_dict.keys():
                slot_value = current_option_dict[slot]
                # This will add 1 to 0 if this is the first time this value has been encountered, or it will add 1
                # to whatever was already in there
                slot_values[slot_value] += 1
        return slot_values

    def count_matches_per_slot(self, current_informs):
        """
        Counts occurrences of each current inform slot (slot and slot_value) in the database items as well as the
        number of matching of database items given the current informs

        For each item in the database and each current inform slot if that slot is in the database item (matches slot
        and slot_value) then increment the count for that slot by 1.

        Parameters:
            current_informs (dict): The current informs/current_inform_slots

        Returns:
            dict: Each slot in current_informs with the count of the number of matches for that slot
        """

        # The items (slot, slot_value) of the current informs are used as a slot to the cached_db_slot
        inform_items = frozenset(current_informs.items())

        # A dict of the inform keys and their counts is stored (or not stored) in the cached_db_slot
        cache_return = self.cached_db_slot[inform_items]
        if cache_return:
            return cache_return

        # Count matches of individual slots
        slot_matches = {key: 0 for key in current_informs.keys()}
        for db_id in self.database.keys():
            for slot, slot_value in current_informs.items():
                # Skip if a no query item and all_slots_match stays true
                if slot in self.no_query:
                    continue
                # If anything all_slots_match stays true AND the specific slot slot gets a +1
                elif slot_value == 'anything':
                    slot_matches[slot] += 1
                elif slot in self.database[db_id].keys():
                    if slot_value.lower() == self.database[db_id][slot].lower():
                        slot_matches[slot] += 1

        # Count full matches of all inform slots
        slot_matches['all_slots'] = len(self.get_matching_db_entries(current_informs))

        # Update cache
        self.cached_db_slot[inform_items].update(slot_matches)
        assert self.cached_db_slot[inform_items] == slot_matches
        return slot_matches
